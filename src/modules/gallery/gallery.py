from typing import Dict
from collections import defaultdict

from modules.templates.templates import sTrackInfo, TrackSession, TimeSession, GalleryElement
from modules.gallery.utils import choose_index

class Gallery:
    def __init__(self, gallery_config: Dict):
        self.max_no_embds = gallery_config.get('max_num_vec', 20)
        self.max_time = gallery_config.get('max_time', 300)
        self.max_live_time = gallery_config.get('max_live_time', 300)
        self.delayT_rmExitedCustomer = gallery_config.get('delayT_rmExitedCustomer', 5)
        self.max_spt_dis = gallery_config.get('max_spt_dis', 400)
        self.customer_gallery: Dict[int, GalleryElement] = {}
        self.num_person = 0
    
    def read_all(self, track_info: sTrackInfo):
        vge = []
        overlap_time_ids = []

        for c_id, ge in self.customer_gallery.items():
            if any(
                min(track_info.time_session.end_time, ses.time_session.end_time) >=
                max(track_info.time_session.start_time, ses.time_session.start_time)
                for ses in ge.sessions
            ):
                overlap_time_ids.append(c_id)

        for c_id, ge in self.customer_gallery.items():
            if c_id in overlap_time_ids:
                continue
            vge.append(ge)

        return vge

    def create_new(self, track_info: sTrackInfo):
        self.num_person += 1
        customer_id = self.num_person
        track_session = TrackSession(track_id=track_info.track_id, time_session=track_info.time_session)
        time_session = TimeSession(start_time=track_info.time_session.start_time, end_time=track_info.time_session.end_time)
        ge = GalleryElement(customer_id=customer_id, sessions=[track_session], embeddings=track_info.embeddings, time_session=time_session)
        self.customer_gallery[customer_id] = ge
        print(f"Create track {track_info.track_id} with cid: {customer_id}.")
        # save reid
        
        return customer_id
    
    # def update_one(self, customer_id, track_info: sTrackInfo):
    #     print(f"Update track {track_info.track_id} camera with cid: {customer_id}.")
        
    #     if customer_id in self.customer_gallery.keys():
    #         track_session = TrackSession(track_info.track_id, track_info.time_session)
    #         for q_embeddings in track_info.embeddings:
    #             g_embeddings = self.customer_gallery[customer_id].embeddings
    #             if len(g_embeddings) > 0:
    #                 self.customer_gallery[customer_id].update_time += 1
    #                 indices = choose_index(
    #                     self.customer_gallery[customer_id].update_time,
    #                     len(g_embeddings),
    #                     len(q_embeddings),
    #                     self.max_no_embds
    #                 )
    #                 print('g_embeddings', g_embeddings)
    #                 print('q_embeddings', q_embeddings)
    #                 all_embeds = g_embeddings + q_embeddings
    #                 self.customer_gallery[customer_id].embeddings = all_embeds[-self.max_no_embds:]
    #                 # self.customer_gallery[customer_id].embeddings = [all_embeds[i] for i in indices]
    #             else:
    #                 self.customer_gallery[customer_id].update_time = 1
    #                 self.customer_gallery[customer_id].embeddings = q_embeddings
            
    #         self.customer_gallery[customer_id].sessions.append(track_session)
    #         self.customer_gallery[customer_id].sessions.sort(key=lambda x: x.time_session.end_time)
    #         if track_info.time_session.start_time < self.customer_gallery[customer_id].time_session.start_time:
    #             self.customer_gallery[customer_id].time_session.start_time = track_info.time_session.start_time
    #         if track_info.time_session.end_time > self.customer_gallery[customer_id].time_session.end_time:
    #             self.customer_gallery[customer_id].time_session.end_time = track_info.time_session.end_time

    def update_one(self, customer_id, track_info: sTrackInfo):
        print(f"Update track {track_info.track_id} camera with cid: {customer_id}.")
        
        if customer_id in self.customer_gallery.keys():
            track_session = TrackSession(track_info.track_id, track_info.time_session)
            
            for q_embeddings in track_info.embeddings:
                g_embeddings = self.customer_gallery[customer_id].embeddings
                
                # Đảm bảo embeddings là list các vector numpy
                if not isinstance(g_embeddings, list):
                    g_embeddings = [g_embeddings]
                if not isinstance(q_embeddings, list):
                    q_embeddings = [q_embeddings]
                
                if len(g_embeddings) > 0:
                    self.customer_gallery[customer_id].update_time += 1
                    indices = choose_index(
                        self.customer_gallery[customer_id].update_time,
                        len(g_embeddings),
                        len(q_embeddings),
                        self.max_no_embds
                    )
                    
                    # Nối 2 list vector
                    all_embeds = g_embeddings + q_embeddings
                    # Giữ lại max_no_embds vector cuối
                    self.customer_gallery[customer_id].embeddings = all_embeds[-self.max_no_embds:]
                else:
                    self.customer_gallery[customer_id].update_time = 1
                    self.customer_gallery[customer_id].embeddings = q_embeddings
            
            self.customer_gallery[customer_id].sessions.append(track_session)
            self.customer_gallery[customer_id].sessions.sort(key=lambda x: x.time_session.end_time)
            
            if track_info.time_session.start_time < self.customer_gallery[customer_id].time_session.start_time:
                self.customer_gallery[customer_id].time_session.start_time = track_info.time_session.start_time
            if track_info.time_session.end_time > self.customer_gallery[customer_id].time_session.end_time:
                self.customer_gallery[customer_id].time_session.end_time = track_info.time_session.end_time

    