# SCR ReID Pipeline

Hệ thống Re-Identification (ReID) hoàn chỉnh sử dụng YOLOv11, SORT tracking, FastReID embedding và gallery matching.

## Cấu trúc dự án

```
sct_reid/
├── assets/                 # Video và ảnh test
├── cfg/                    # File cấu hình
├── src/                    # Source code chính
│   ├── main.py            # Pipeline chính
│   ├── test_pipeline.py   # Test pipeline
│   └── modules/           # Các module
│       ├── detection/     # YOLOv11 detection
│       ├── tracker/       # SORT tracking
│       ├── track_manager/ # Quản lý track
│       ├── gallery/       # Quản lý gallery
│       ├── matching/      # Matching algorithm
│       └── embedding/     # FastReID embedding
└── weights/               # Model weights
```

## Pipeline hoạt động

### 1. Detection (YOLOv11)
- Phát hiện người trong frame
- Trả về bounding boxes

### 2. Tracking (SORT)
- Theo dõi đối tượng qua các frame
- Quản lý track ID

### 3. Track Management
- Quản lý thông tin track
- Trích xuất embedding khi track kết thúc
- Join tracks nếu cần

### 4. Gallery Management
- Lưu trữ thông tin customer
- Quản lý embedding vectors
- Cập nhật gallery với track mới

### 5. Matching
- So khớp track mới với gallery
- Sử dụng cosine distance
- Threshold-based matching

## Cách sử dụng

### 1. Chạy pipeline chính
```bash
cd src
python main.py
```

### 2. Test pipeline
```bash
cd src
python test_pipeline.py
```

### 3. Test matching
```bash
cd src
python test.py
```

### 4. Test matching fix
```bash
cd src
python test_matching_fix.py
```

### 5. Test simple pipeline
```bash
cd src
python test_simple.py
```

## Cấu hình

File cấu hình chính: `cfg/cfg.yaml`

### Các tham số quan trọng:

#### Detection (YOLOv11)
- `conf_thresh`: 0.3 - Confidence threshold
- `iou_thresh`: 0.3 - IoU threshold
- `classes`: [0] - Chỉ detect person

#### Tracking (SORT)
- `max_age`: 15 - Tuổi tối đa của track
- `min_hits`: 2 - Số frame tối thiểu để init track

#### Track Manager
- `min_num_embeds`: 3 - Số embedding tối thiểu
- `max_num_embeds`: 32 - Số embedding tối đa
- `max_fragment_frame`: 30 - Frame tối đa để giữ track

#### Gallery
- `max_no_embds`: 20 - Số embedding tối đa per customer
- `max_time`: 300 - Thời gian tối đa (giây)

#### Matching
- `threshold`: 0.3 - Distance threshold cho matching

## Output

### Video output
- File: `output/reid_output.mp4`
- Hiển thị bounding boxes và track ID
- Thống kê real-time

### Console output
- Thống kê processing
- Matching results
- Gallery updates

## Thành phần chính

### 1. SCRReid class (main.py)
- Pipeline chính
- Quản lý toàn bộ workflow
- Visualization và output

### 2. TrackManager
- Quản lý track lifecycle
- Trích xuất embedding
- Join tracks

### 3. Gallery
- Lưu trữ customer information
- Quản lý embedding vectors
- Time-based filtering

### 4. Matching
- Cosine distance calculation
- Threshold-based matching
- Frequency-based ranking

## Dependencies

```
opencv-python
numpy
torch
ultralytics
scipy
yacs
pyyaml
```

## Lưu ý

1. Đảm bảo có đủ RAM cho video processing
2. GPU recommended cho embedding extraction
3. Điều chỉnh threshold theo use case
4. Monitor gallery size để tránh memory overflow

## Troubleshooting

### Lỗi thường gặp:
1. **Video không mở được**: Kiểm tra đường dẫn file
2. **Model không load**: Kiểm tra weights path
3. **Memory error**: Giảm max_num_embeds
4. **Slow performance**: Sử dụng GPU cho embedding

### Debug mode:
```python
# Trong main.py, thêm debug prints
print(f"Debug: {variable}")
``` 