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
        
        
    def run(self):
        frame_count = 0
        start_time = datetime.datetime.strptime('2015-08-02 16:00:00', "%Y-%m-%d %H:%M:%S")

        print("Starting ReID pipeline...")
        print(f"Video FPS: {self.fps}")
        print("=" * 50)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            cur_time = start_time + datetime.timedelta(seconds=frame_count/self.fps)
            # print(f'[INFO] Processing frame {frame_count}')
            l_bboxes = self.detector.detect(frame)
            dets = np.array([list(map(int, bbox)) for bbox in l_bboxes])
            alive_tracks, alive_indices, dead_tracks = self.tracker.update(dets)
            self.track_manager.update_session_tracks(alive_tracks, dead_tracks, frame, frame_count, cur_time)
            for track in alive_tracks:
                x1, y1, x2, y2, _, track_id = map(int, track)

                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                # Vẽ ID người
                text = f"ID: {track_id}"
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
            
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    config = load_config('../cfg/cfg.yaml')
    sct_reid = SCRReid(config)
    sct_reid.run()
            
