from typing import Dict, List
from dataclasses import dataclass, field
import datetime

class DetectionBox:
    def __init__(self, bbox: List[float], box_id: int, iou: float):
        self.bbox = bbox
        self.box_id = box_id
        self.iou = iou
    
    def to_dict(self) -> Dict:
        return {
            'bbox': self.bbox,
            'box_id': self.box_id,
            'iou': self.iou
        }

class TrackInfo:
    def __init__(self, track_id: int, frame_id: int, timestamp: datetime.datetime):
        self.track_id = track_id
        self.cropped_person: List[List[float]] = []
        self.embeddings: List[List[float]] = []
        self.bboxes: List[List[float]] = []
        
        self.start_frame = frame_id
        self.end_frame = frame_id
        
        self.start_time = timestamp
        self.end_time = timestamp
        
        self.is_inited = False
        self.is_dead = False
    
    def get_len_embeddings(self):
        return len(self.embeddings)
    
    @property
    def get_len_bboxes(self):
        return len(self.bboxes)
    
    def to_dict(self) -> Dict:
        return {
            'track_id': self.track_id,
            'cropped_person': self.cropped_person,
            'embeddings': self.embeddings,
            'bboxes': self.bboxes,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'is_inited': self.is_inited,
            'is_dead': self.is_dead,
            'len_embeddings': self.get_len_embeddings(),
            'len_bboxes': self.get_len_bboxes
        }
        
    

@dataclass
class TimeSession:
    start_time: datetime
    end_time: datetime

@dataclass
class TrackSession:
    track_id: int
    time_session: TimeSession
        
@dataclass
class sTrackInfo(TrackSession):
    embeddings: List[List[float]] = field(default_factory=list)
    
    def has_embeddings(self):
        if len(self.embeddings) > 0:
            return True
        return False
        
    def get_all_vecs(self):
        return self.embeddings

@dataclass
class GalleryElement:
    customer_id: int
    sessions: List[TrackSession]
    embeddings: List[List[float]]
    time_session: TimeSession
    update_time: int = None
         
    def __post_init__(self):
        self.update_time = 1 if self.update_time is None else self.update_time

@dataclass
class MatchingResult:
    """A class template that presents matching result.
    """
    match_id: int
    match_frequency: float
    match_distance: float