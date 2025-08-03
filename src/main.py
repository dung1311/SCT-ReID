import cv2
import numpy as np
import datetime
from typing import Dict, List

from modules.config_loader.yaml_loader import load_config
from modules.detection.yolov11 import Yolov11Detector
from modules.tracker.sort import Sort
from modules.track_manager.track_manager import TrackManager
from modules.gallery.gallery import Gallery
from modules.templates.templates import sTrackInfo, TimeSession, TrackInfo
from modules.matching.matching import Matching


class SCRReid:
    def __init__(self, config: Dict):
        self.tracker = Sort(config["Tracking"]["box_track"])
        self.detector = Yolov11Detector(config["DETECTION"]["yolov11"])
        self.track_manager = TrackManager(config["TRACK_MANAGER"])
        self.gallery = Gallery(config["GALLERY"])
        self.matching = Matching(config["TRACK_MANAGER"]["join_track"])
        
        # Video capture
        self.cap = cv2.VideoCapture('../assets/video_2min.mp4')
        self.fps = 30
        
        # ReID configuration
        self.reid_config = config["TRACK_MANAGER"]
        self.min_track_length = self.reid_config.get("min_track_length_for_reid", 10)
        self.reid_threshold = self.reid_config.get("reid_threshold", 0.7)
        
        # Visualization
        self.show_display = config.get("DISPLAY", {}).get("show", True)
        self.save_results = config.get("OUTPUT", {}).get("save", False)
        
        if self.save_results:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter('../output/reid_output.avi', fourcc, self.fps, (1920, 1080))
        
    def run(self):
        frame_count = 0
        start_time = datetime.datetime.strptime('2015-08-02 16:00:00', "%Y-%m-%d %H:%M:%S")

        print("Starting ReID pipeline...")
        print(f"Video FPS: {self.fps}")
        print(f"Reid threshold: {self.reid_threshold}")
        print(f"Min track length for reid: {self.min_track_length}")
        print("=" * 50)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            cur_time = start_time + datetime.timedelta(seconds=frame_count/self.fps)
            
            if frame_count % 30 == 0:  # Log every second
                print(f"Processing Frame: {frame_count} | Time: {cur_time.strftime('%H:%M:%S')}")

            # Step 1: Object Detection
            l_boxes = self.detector.detect(frame)
            dets = np.array([list(map(int, bbox)) for bbox in l_boxes]) if l_boxes else np.empty((0, 4))

            # Step 2: Object Tracking  
            alive_tracks, alive_indices, dead_track_ids = self.tracker.update(dets)
            
            # Step 3: Update track manager with current frame info
            self.track_manager.update_session_tracks(alive_tracks, dead_track_ids, frame, frame_count, cur_time)
            
            # Step 4: Process tracks for ReID
            self.process_tracks_for_reid(frame_count, cur_time)
            
            # Step 5: Visualization
            if self.show_display or self.save_results:
                annotated_frame = self.draw_results(frame, alive_tracks)
                
                if self.show_display:
                    cv2.imshow('Person ReID', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                if self.save_results:
                    self.out.write(annotated_frame)

        # Cleanup
        self.cleanup()
        
        # Final statistics
        self.print_final_statistics()

    def process_tracks_for_reid(self, frame_count: int, cur_time: datetime.datetime):
        """Process completed tracks for person re-identification"""
        
        # Get all tracks that are ready for ReID processing
        ready_tracks = self.get_tracks_ready_for_reid()
        
        for track_info in ready_tracks:
            if not track_info.embeddings:
                continue
                
            # Convert TrackInfo to sTrackInfo for gallery processing
            strack_info = self.convert_to_strack_info(track_info, cur_time)
            
            # Search in gallery for matching person
            candidate_gallery_elements = self.gallery.read_all(strack_info)
            
            if candidate_gallery_elements:
                # Find best match
                match_results = self.matching.match_with_all_ids(strack_info, candidate_gallery_elements)
                best_match = self.find_best_match(match_results)
                
                if best_match and best_match.match_distance < self.reid_threshold:
                    # Update existing customer
                    customer_id = best_match.match_id
                    self.gallery.update_one(customer_id, strack_info)
                    print(f"✓ Track {track_info.track_id} matched with Customer {customer_id} "
                          f"(dist: {best_match.match_distance:.3f}, freq: {best_match.match_frequency})")
                else:
                    # Create new customer
                    customer_id = self.gallery.create_new(strack_info)
                    print(f"✓ Track {track_info.track_id} registered as new Customer {customer_id}")
            else:
                # No candidates in gallery, create new customer
                customer_id = self.gallery.create_new(strack_info)
                print(f"✓ Track {track_info.track_id} registered as new Customer {customer_id}")
    
    def get_tracks_ready_for_reid(self) -> List[TrackInfo]:
        """Get tracks that are dead and ready for ReID processing"""
        ready_tracks = []
        
        for track_id, track_info in self.track_manager.dict_tracks.items():
            if (track_info.is_dead and 
                track_info.is_inited and 
                len(track_info.bboxes) >= self.min_track_length and
                track_info.embeddings and
                not hasattr(track_info, 'processed_for_reid')):
                
                # Mark as processed to avoid reprocessing
                track_info.processed_for_reid = True
                ready_tracks.append(track_info)
        
        return ready_tracks
    
    def convert_to_strack_info(self, track_info: TrackInfo, cur_time: datetime.datetime) -> sTrackInfo:
        """Convert TrackInfo to sTrackInfo format expected by Gallery"""
        time_session = TimeSession(
            start_time=track_info.start_time,
            end_time=track_info.end_time
        )
        
        strack_info = sTrackInfo(
            track_id=track_info.track_id,
            time_session=time_session,
            embeddings=track_info.embeddings.copy()
        )
        
        return strack_info
    
    def find_best_match(self, match_results):
        """Find the best matching result based on distance and frequency"""
        if not match_results:
            return None
            
        # Filter matches with minimum frequency
        min_frequency = 3
        valid_matches = [m for m in match_results if m.match_frequency >= min_frequency]
        
        if not valid_matches:
            return None
            
        # Return the match with highest frequency, then lowest distance
        return valid_matches[0]  # Already sorted by frequency desc, distance asc
    
    def draw_results(self, frame, alive_tracks):
        """Draw bounding boxes and track IDs on frame"""
        annotated_frame = frame.copy()
        
        # Draw active tracks
        for track in alive_tracks:
            track_id = int(track[-1])
            bbox = list(map(int, track[:4]))
            x1, y1, x2, y2 = bbox
            
            # Get customer ID if available
            customer_id = self.get_customer_id_for_track(track_id)
            
            # Draw bounding box
            color = self.get_color_for_track(track_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track info
            label = f"T:{track_id}"
            if customer_id is not None:
                label += f" C:{customer_id}"
                
            cv2.rectangle(annotated_frame, (x1, y1-25), (x1+len(label)*8, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1+2, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw statistics
        stats_text = [
            f"Active Tracks: {len(alive_tracks)}",
            f"Total Customers: {self.gallery.num_person}",
            f"Gallery Size: {len(self.gallery.customer_gallery)}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(annotated_frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame
    
    def get_customer_id_for_track(self, track_id):
        """Get customer ID associated with a track ID"""
        for customer_id, ge in self.gallery.customer_gallery.items():
            for session in ge.sessions:
                if session.track_id == track_id:
                    return customer_id
        return None
    
    def get_color_for_track(self, track_id):
        """Generate consistent color for track ID"""
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        if hasattr(self, 'out'):
            self.out.release()
        cv2.destroyAllWindows()
        print("\nPipeline completed successfully!")
    
    def print_final_statistics(self):
        """Print final statistics"""
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Total unique customers identified: {self.gallery.num_person}")
        print(f"Customers in gallery: {len(self.gallery.customer_gallery)}")
        
        # Print customer sessions
        for customer_id, ge in self.gallery.customer_gallery.items():
            session_count = len(ge.sessions)
            start_time = ge.time_session.start_time.strftime('%H:%M:%S')
            end_time = ge.time_session.end_time.strftime('%H:%M:%S')
            embedding_count = len(ge.embeddings)
            
            print(f"Customer {customer_id}: {session_count} sessions, "
                  f"{start_time}-{end_time}, {embedding_count} embeddings")
        
        print("="*50)


def main():
    """Main function to run the ReID pipeline"""
    try:
        # Load configuration
        config = load_config('../cfg/cfg.yaml')
        
        # Initialize and run pipeline
        reid_system = SCRReid(config)
        reid_system.run()
        
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found - {e}")
    except Exception as e:
        print(f"Error running ReID pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()