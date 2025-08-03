import datetime
from modules.matching.matching import Matching
from modules.templates.templates import *

if __name__ == '__main__':
    # Config threshold = 0.5 để lọc cosine distance nhỏ hơn 0.5
    matching = Matching({'threshold': 0.5})
    
    # Tạo query giả lập
    query = TrackInfo(track_id=1, timestamp=datetime.datetime.now(), frame_id=1)
    query.embeddings = [
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4]
    ]
    
    # Tạo nhiều gallery giả lập
    galleries = [
        GalleryElement(customer_id=101, embeddings=[
            [0.1, 0.2, 0.31],
            [0.9, 0.1, 0.0],
            [0.2, 0.1, 0.4]
        ], sessions=None, time_session=None),

        GalleryElement(customer_id=102, embeddings=[
            [0.5, 0.5, 0.5],
            [0.2, 0.1, 0.39]
        ], sessions=None, time_session=None),

        GalleryElement(customer_id=103, embeddings=[
            [0.1, 0.21, 0.31],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2]
        ], sessions=None, time_session=None),

        GalleryElement(customer_id=104, embeddings=[
            [0.9, 0.9, 0.9],
            [0.8, 0.8, 0.8]
        ], sessions=None, time_session=None)
    ]
    
    # Chạy thử match từng gallery
    print("=== Kết quả _match_with_one_id cho từng gallery ===")
    for g in galleries:
        result = matching._match_with_one_id(query, g)
        print(f"Gallery {g.customer_id}: {result}")

    # Nếu muốn xem tất cả cùng lúc và sort theo độ match
    print("\n=== Kết quả match_with_all_ids (đã sort) ===")
    results_all = matching.match_with_all_ids(query, galleries)
    for r in results_all:
        print(r)
        
