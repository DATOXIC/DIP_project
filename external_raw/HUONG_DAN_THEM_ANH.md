# HƯỚNG DẪN THÊM ẢNH TỪ MẠNG / ĐIỆN THOẠI VÀO DATASET
=====================================================

Thư mục này chứa ảnh thô tải từ mạng hoặc chụp từ điện thoại cho 5 ký hiệu:

```
c:/DIP/DIP_project/external_raw/
  ├── A/   <-- Thả ảnh chữ 'A' vào đây (.jpg, .png, .jpeg, .webp)
  ├── B/   <-- Thả ảnh chữ 'B' vào đây
  ├── C/   <-- Thả ảnh chữ 'C' vào đây
  ├── L/   <-- Thả ảnh chữ 'L' vào đây
  └── V/   <-- Thả ảnh chữ 'V' vào đây
```

---

### Các bước thực hiện:

1. **Thả ảnh vào thư mục tương ứng:**
   * Ví dụ: Tải 10 ảnh chữ A trên mạng về, copy thả vào thư mục `external_raw/A/`.
   * Tên file ảnh đặt tuỳ ý (ví dụ: `image1.jpg`, `download_02.png`...).

2. **Chạy công cụ chuyển đổi sang Tensor DIP:**
   Mở terminal gõ:
   ```bash
   py legacy/process_external_images.py
   ```
   *Muốn xem trước trực quan từng ảnh được bóc tách viền và CLAHE ra sao:*
   ```bash
   py legacy/process_external_images.py --preview
   ```

3. **Huấn luyện mô hình:**
   Sau khi chuyển đổi xong, toàn bộ mẫu đã được lưu vào `vsl_dataset/samples/`. Chỉ cần chạy:
   ```bash
   py train.py
   ```
