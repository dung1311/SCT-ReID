from typing import Dict, List
from scipy.spatial.distance import cdist

from modules.templates.templates import MatchingResult, TrackInfo, GalleryElement

METRIC = 'cosine'

class Matching:
    def __init__(self, matching_config: Dict):
        self.distance_threshold = matching_config['threshold']
    
    def match_with_all_ids(self, query_track: TrackInfo, l_ge: List[GalleryElement]) -> List[MatchingResult]:
        match_results = [self._match_with_one_id(query_track, ge) for ge in l_ge]
        match_results.sort(key=lambda x: (x.match_frequency, -x.match_distance), reverse=True)
        
        return match_results
    
    def _match_with_one_id(self, query_track: TrackInfo, l_ge: GalleryElement) -> MatchingResult:
        match_res = MatchingResult(match_id=-1, match_frequency=0, match_distance=0.0)
        
        q_embedds = query_track.embeddings
        ge_embedds = l_ge.embeddings
        
        # calculate distance
        match_res = self.calculate_distance(q_embedds, ge_embedds)
        match_res.match_id = l_ge.customer_id
        
        return match_res
    
    def calculate_distance(self, q_embedds, ge_embedds) -> MatchingResult:
        match_results = MatchingResult(match_id=-1, match_frequency=0, match_distance=0.0)
        if q_embedds and ge_embedds:
            dist_mtx = cdist(q_embedds, ge_embedds, metric=METRIC)
            masked_dist_mtx = dist_mtx[dist_mtx < self.distance_threshold]
            match_frequency = len(masked_dist_mtx)
            mean_dist = masked_dist_mtx.mean() if match_frequency > 0 else dist_mtx.mean()
            match_results.match_frequency = match_frequency
            match_results.match_distance = mean_dist
        
        return match_results

    
        