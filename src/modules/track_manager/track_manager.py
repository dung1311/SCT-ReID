from typing import Dict, List

from modules.templates.templates import TrackInfo, TimeSession, sTrackInfo
from modules.matching.matching import Matching
from modules.gallery.gallery import Gallery
from modules.embedding.fastreid.embed import Embedding

class TrackManager:
    def __init__(self, track_manager_config: Dict):
        self.dict_tracks: Dict[int, TrackInfo] = {}
        self.config = track_manager_config
        self.select_cfg = track_manager_config["VECTORIZATION"]["selection"]
        self.num_init = track_manager_config["min_hits"]
        self.is_join_track = track_manager_config["join_track"]["enable"]
        
        vec_config = track_manager_config["VECTORIZATION"]["fast_reid"]
        self.embed = Embedding(vec_config)
        
        if self.is_join_track:
            self.max_dist_join = track_manager_config["join_track"]["max_dist"]
            self.matching = Matching(track_manager_config["join_track"])
            self._joined_tracks: Dict[int, TrackInfo] = {}
            self.gallery = Gallery(track_manager_config["GALLERY"])
            
        
        self.num_init = track_manager_config["min_hits"] 
        self.min_num_embeds = track_manager_config["min_num_embeds"]
        self.max_num_embeds = track_manager_config["max_num_embeds"]
        self.max_fragment_frame = track_manager_config["max_fragment_frame"]
        
    def add_new_track_to_dict(self, track_id, new_track: TrackInfo):
        self.dict_tracks[track_id] = new_track
    
    def update_session_tracks(self, alive_tracks: List, dead_tracks: List[int], frame, frame_id, timestamp):
        for alive_track in alive_tracks:
            track_id = int(alive_track[-1])
            bbox = list(map(int, alive_track[:4]))
            # track not in inited tracks
            if track_id not in self.dict_tracks.keys():
                new_track = TrackInfo(track_id, frame_id, timestamp)
                self.add_new_track_to_dict(track_id, new_track)
            
            # update values
            cropped_person = crop_box(frame, bbox)
            self.dict_tracks[track_id].cropped_person.append(cropped_person)
            self.dict_tracks[track_id].bboxes.append(bbox)
            self.dict_tracks[track_id].embeddings.append(self.embed.extract_feature(cropped_person))
            self.dict_tracks[track_id].end_frame = frame_id
            self.dict_tracks[track_id].end_time = timestamp
            
            # if track is inited
            if not self.dict_tracks[track_id].is_inited and len(self.dict_tracks[track_id].bboxes) >= self.num_init:
                match_track_id = -1
                if self.is_join_track:
                    self.update_embedding_of_track(track_id)
                    match_track_id = self.join_tracks(self.dict_tracks[track_id])

                if match_track_id != -1:
                    self.recover_tracks(match_track_id, self.dict_tracks[track_id])
                    # alive_track.track_id = match_track_id
                else:
                    self.dict_tracks[track_id].is_inited = True
                    print(f"Init new track ID {self.dict_tracks[track_id].track_id}")
        
        for dead_track_id in dead_tracks:
            if dead_track_id < 0: 
                continue
            if dead_track_id in self.dict_tracks.keys():
                self.update_embedding_of_track(dead_track_id)
                self.dict_tracks[dead_track_id].is_dead = True
        
        # remove tracks with max fragment frame
        remove_tracks = []
        for track_id, track_info in self.dict_tracks.items():
            if track_info.is_dead and frame_id - track_info.end_frame > self.max_fragment_frame:
                remove_tracks.append(track_id)
        
        for track_id in remove_tracks:
            self.dict_tracks.pop(track_id)
            if self.is_join_track and track_id in self._joined_tracks:
                self._joined_tracks.pop(track_id)
        
        return self.dict_tracks

    def update_embedding_of_track(self, track_id: int):
        selected_crop_persons = []
        if len(self.dict_tracks[track_id].embeddings) >= self.max_num_embeds:
            selected_crop_persons = self.dict_tracks[track_id].cropped_person[-self.max_num_embeds:]
        if not selected_crop_persons:
            return
        
        embeddings = self.embed.extract_feature(selected_crop_persons)
        for embedding in embeddings:
            self.dict_tracks[track_id].embeddings.append(embedding)

        print("Check len embed:", self.dict_tracks[track_id].get_len_embeddings())
    
    def join_tracks(self, new_track: TrackInfo):
        candidate_tracks = []
        for track_info in self.dict_tracks.values():
            self.gallery.customer_gallery.clear()
            if not track_info.is_dead:
                continue
            start_time = track_info.start_time
            end_time = track_info.end_time
            time_session = TimeSession(start_time, end_time)
            embeddings = track_info.embeddings
            s_track_info = sTrackInfo(track_id=track_info.track_id, time_session=time_session, embeddings=embeddings)
            customer_id = self.gallery.create_new(s_track_info)
            self.gallery.update_one(customer_id, s_track_info)
            candidate_tracks.append(self.gallery.customer_gallery[customer_id])
        
        match_results = self.matching.match_with_all_ids(new_track, candidate_tracks)
        
        for match_result in match_results:
            if match_result.match_frequency >= 5:
                print(f"Join track ID {new_track.track_id} with {match_result.match_id} distance {match_result.match_distance} and freq {match_result.match_frequency}")
                return match_result.match_id
        
        return -1
    
    def recover_tracks(self, match_track_id: int, new_track: TrackInfo):
        # add new_track to joined tracks
        self._joined_tracks[match_track_id] = new_track
        
        # recover track
        self.dict_tracks[match_track_id].is_dead = False
        self.dict_tracks[match_track_id].bboxes.extend(self.dict_tracks[new_track.track_id].bboxes)
        self.dict_tracks[match_track_id].cropped_person.extend(self.dict_tracks[new_track.track_id].cropped_person)
        # remove track
        self.dict_tracks.pop(new_track.track_id)
    
def crop_box(frame, bbox):
    x1, y1, x2, y2 = bbox
    cropped_frame = frame[y1:y2, x1:x2]
    return cropped_frame

def check_condition(box, box_iou, select_cfg):        
    x1, y1, x2, y2 = box 
    w = x2 - x1
    h = y2 - y1
    if box_iou < select_cfg['box_iou'] and min(h, w) >= select_cfg['min_size'] and h >= select_cfg['box_ratio'] * w:
        return True
    else:
        return False
