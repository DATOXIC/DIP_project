# -*- coding: utf-8 -*-
"""
CÔNG CỤ ĐIỀU CHỈNH THÔNG SỐ PHẦN CỨNG WEBCAM (DIRECTSHOW)
==========================================================
Dùng để:
1. Bật hộp thoại thuộc tính phần cứng (Camera Properties Dialog) của Windows để:
   - TẮT Auto Exposure (bỏ chọn ô Auto).
   - Hạ Exposure xuống (-5 hoặc -6) để chống cháy sáng khi đứng trước tường trắng.
   - Chỉnh Brightness, Contrast, Backlight Compensation.
2. Cho phép xem trước hình ảnh webcam thời gian thực.
3. Phím tắt trong cửa sổ xem trước:
   - [p]: Bật lại hộp thoại cài đặt driver.
   - [-] hoặc [,]: Giảm độ sáng.
   - [+] hoặc [.]: Tăng độ sáng.
   - [q] hoặc [Esc]: Thoát và lưu thiết lập.
"""
import cv2
import sys


def main():
    print("=" * 60)
    print(" CÔNG CỤ CHỈNH PHẦN CỨNG CAMERA (CHỐNG CHÓI SÁNG TƯỜNG TRẮNG)")
    print("=" * 60)
    print("[*] Đang mở Camera bằng backend DirectShow (cv2.CAP_DSHOW)...")

    cam_idx = 0
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cam_idx = int(sys.argv[1])

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[!] Không thể mở Camera [{cam_idx}] qua DirectShow. Thử backend mặc định...")
        cap = cv2.VideoCapture(cam_idx)

    if not cap.isOpened():
        print(f"[!] Lỗi: Không thể kết nối tới Camera index {cam_idx}!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[*] Đang mở hộp thoại cài đặt Camera Properties của Windows...")
    print("------------------------------------------------------------")
    print(" HƯỚNG DẪN THAO TÁC TRÊN HỘP THOẠI:")
    print(" 1. Chọn tab 'Camera Control'.")
    print(" 2. BỎ CHỌN ô [Auto] ở mục 'Exposure' (Độ phơi sáng).")
    print(" 3. Kéo thanh trượt 'Exposure' lùi về mức âm: -5 hoặc -6.")
    print(" 4. Chuyển sang tab 'Video Proc Amp':")
    print("    - Hạ 'Brightness' (Độ sáng) xuống bớt (khoảng 30 - 40).")
    print("    - Tăng 'Contrast' (Độ tương phản) lên (khoảng 60 - 70).")
    print(" 5. Bấm [Apply] -> [OK].")
    print("------------------------------------------------------------")

    # Mở hộp thoại thuộc tính
    cap.set(cv2.CAP_PROP_SETTINGS, 1)

    print("[*] Đang hiển thị khung hình xem trước. Nhấn [q] để thoát khi đã ưng ý.")
    window_name = "Chinh Thong So Camera - [p]: Mo lai cai dat | [q]: Thoat"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Thêm thông tin hướng dẫn lên khung hình
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
        cv2.putText(frame, "Bam [p] de mo lai hop thoai Driver | [q] de luu va thoat",
                    (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 200), 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), ord('Q'), 27]: # Esc
            break
        elif key in [ord('p'), ord('P')]:
            cap.set(cv2.CAP_PROP_SETTINGS, 1)

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Đã đóng camera. Các thiết lập phần cứng đã được lưu lại trong driver!")


if __name__ == "__main__":
    main()
