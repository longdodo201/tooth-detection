# Audit annotation — quadrant_enumeration

## Phạm vi và nguồn dữ liệu

- JSON: `D:\long\training_data\quadrant_enumeration\train_quadrant_enumeration.json`
- Ảnh: `D:\long\training_data\quadrant_enumeration\xrays\`
- Repository: `D:\long\tooth-detection`; audit artifacts chỉ nằm trong `reports/annotation_audit/`.
- Phân tích dựa trên JSON, kiểm tra hình học và overlay trực quan cho 16 ảnh ưu tiên. Không dùng test set.

## Toàn bộ dataset

- Ảnh JSON/PNG: 634/634; ảnh thiếu: 0; PNG thừa: 0.
- Annotation: 18095; annotation tham chiếu ảnh không tồn tại: 0.
- Duplicate image IDs: 0; duplicate annotation IDs: 0; category ID không định nghĩa: 0.
- Ảnh kích thước khác PNG header: 0; bbox/polygon/geometry issues: 0; bbox–polygon envelope sai lệch >2 px: 0 (max 1.00 px).
- Polygon có điểm lặp: 8; polygon suy biến: 0; bbox/polygon vượt biên: 0.
- Annotation hình học trùng chính xác: 0 cặp.
- `category_id_1` counts: `{0: 4506, 1: 4490, 2: 4542, 3: 4557}`; `category_id_2` counts: `{0: 2508, 1: 2497, 2: 2499, 3: 2404, 4: 2299, 5: 2070, 6: 2314, 7: 1504}`.
- Cặp category hiện diện: 32/32; số annotation mỗi cặp min–max: 341–630.

## Các ảnh ưu tiên

- Ảnh tìm thấy: 16/16.
- Số ảnh theo trạng thái: `{'NEEDS_EXPERT_REVIEW': 16}`.
- Cặp lặp (trong 16 ảnh): 37; annotation vượt số cặp duy nhất: 37.
- `SUSPECTED_DUPLICATE` là cờ sàng lọc, không phải kết luận annotation sai. `NEEDS_EXPERT_REVIEW` phản ánh thiếu định nghĩa taxonomy/ground truth chuyên môn.
- Các IoU polygon trong CSV được tính bằng raster mask ở độ phân giải pixel cục bộ; bbox IoU là phép tính liên tục theo bbox.

## Kết luận

- Kiểm tra tự động xác nhận phần lớn cấu trúc bbox/polygon hợp lệ; không tự sửa hoặc loại annotation.
- Overlay giúp quan sát các polygon gần viền răng, nhưng không đủ để kết luận lâm sàng hoặc xác thực toàn bộ dataset.
- Các cặp nhãn lặp có thể là nhiều đối tượng hợp lệ hoặc vấn đề annotation; cần xem theo từng ảnh và xác nhận quy ước với người tạo nhãn/chuyên gia.
- Không ánh xạ sang FDI.

## Tệp đầu ra

- `image_findings.csv` — tổng hợp 16 ảnh.
- `annotation_findings.csv` — annotation nằm trong cặp lặp hoặc có overlap đáng xem.
- `overlays/` — overlay JPEG cho từng ảnh ưu tiên.
- `limitations.md` — giới hạn và nội dung cần chuyên gia xác nhận.
