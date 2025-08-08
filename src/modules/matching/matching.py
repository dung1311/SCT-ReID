from typing import Dict, List
from scipy.spatial.distance import cdist
import numpy as np

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
    
    # def calculate_distance(self, q_embedds, ge_embedds) -> MatchingResult:
    #     match_results = MatchingResult(match_id=-1, match_frequency=0, match_distance=0.0)
    #     if len(q_embedds) and len(ge_embedds):
    #         print('q_embeds', q_embedds)
    #         print('ge_embeds', ge_embedds)
    #         dist_mtx = cdist(q_embedds, ge_embedds, metric=METRIC)
    #         masked_dist_mtx = dist_mtx[dist_mtx < self.distance_threshold]
    #         match_frequency = len(masked_dist_mtx)
    #         mean_dist = masked_dist_mtx.mean() if match_frequency > 0 else dist_mtx.mean()
    #         match_results.match_frequency = match_frequency
    #         match_results.match_distance = mean_dist
        
    #     return match_results

    def calculate_distance(self, q_embedds, ge_embedds) -> MatchingResult:
        """
        Convert inputs to 2-D numpy arrays safely, then compute distance matrix.
        Returns MatchingResult with match_frequency and match_distance.
        """
        def _to_2d_array(x):
            """Convert x (list/ndarray/nested) to a 2-D numpy array with shape (N, D)."""
            if x is None:
                return np.zeros((0, 0), dtype=float)

            # Quick path: already numpy array
            arr = np.asarray(x)
            # squeeze trivial extra dims (e.g. shape (1, D) or (1,1,D))
            arr = np.squeeze(arr)

            if arr.ndim == 0:
                # single scalar -> treat as 1x1
                return arr.reshape(1, 1).astype(float)
            if arr.ndim == 1:
                # single vector -> (1, D)
                return arr.reshape(1, -1).astype(float)
            if arr.ndim == 2:
                return arr.astype(float)

            # If arr has >2 dims (rare), try to stack elements (handles list of vectors)
            try:
                stacked = np.vstack([np.asarray(e).squeeze() for e in x])
                if stacked.ndim == 1:
                    return stacked.reshape(1, -1).astype(float)
                return stacked.astype(float)
            except Exception as e:
                raise ValueError(f"Cannot convert embeddings to 2D array: {e}")

        match_results = MatchingResult(match_id=-1, match_frequency=0, match_distance=0.0)

        # convert inputs
        try:
            q = _to_2d_array(q_embedds)
            g = _to_2d_array(ge_embedds)
        except ValueError as e:
            # optional: log the error for debugging
            print("convert embeddings error:", e)
            return match_results

        # debug: show shapes (helps track down bad inputs)
        print("calculate_distance: q.shape =", q.shape, "g.shape =", g.shape)

        # if either empty -> nothing to match
        if q.size == 0 or g.size == 0:
            return match_results

        # ensure dimension compatibility
        if q.shape[1] != g.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: q has dim {q.shape[1]}, g has dim {g.shape[1]}")

        # now safe to call cdist
        dist_mtx = cdist(q, g, metric=METRIC)  # shape (num_q, num_g)

        # mask of distances below threshold
        match_mask = dist_mtx < self.distance_threshold
        match_frequency = int(match_mask.sum())

        # compute match distance:
        # - if we have any below-threshold distances, average them;
        # - otherwise use the minimum distance (closest attempt)
        if match_frequency > 0:
            mean_dist = float(dist_mtx[match_mask].mean())
        else:
            mean_dist = float(dist_mtx.min())

        match_results.match_frequency = match_frequency
        match_results.match_distance = mean_dist

        return match_results
