# Giới hạn và yêu cầu xem xét chuyên môn

- Không tìm thấy data dictionary, protocol gán nhãn hoặc ánh xạ category sang FDI trong repo và các thư mục dataset đã kiểm tra.
- Overlay chỉ quan sát 16 ảnh ưu tiên; không kiểm tra thủ công toàn bộ 18.095 annotation.
- Polygon có hình dạng gần viền răng trong ảnh mẫu, nhưng chỉ nha sĩ/nhóm gán nhãn có thể xác nhận đối tượng và độ đúng lâm sàng.
- Cặp category lặp không mặc nhiên là lỗi. Các cặp có IoU thấp có thể biểu thị nhiều đối tượng; các cặp có IoU cao chỉ là ứng viên trùng cần người xem.
- Các trường hợp có cùng cặp category xuất hiện ở nhiều vị trí cần xác nhận cách định nghĩa index, răng hỗn hợp/đối tượng chồng hình và xử lý nhiều tooth instance trong cùng vị trí.
- IoU polygon được ước lượng qua rasterization pixel bằng Pillow; sai số nhỏ do làm tròn tọa độ có thể có.
- Không xem xét test set để tối ưu nhãn hoặc mô hình; không đánh giá model.
