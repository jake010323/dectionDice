# -*- coding: utf-8 -*-
import os
import cv2
import pytesseract
import pandas as pd
from unidecode import unidecode

# Thư mục chứa ảnh CCCD
FOLDER_PATH = "cccd_images"  # hoặc đường dẫn tuyệt đối nếu cần

# Hàm xử lý một ảnh CCCD
def process_image(image_path):
    try:
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không đọc được ảnh: {image_path}")
            return {}

        # Tiền xử lý ảnh (chuyển về xám, tăng độ tương phản, v.v.)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Nhận diện văn bản
        text = pytesseract.image_to_string(gray, lang='vie')

        # Chuẩn hóa văn bản
        text_no_diacritics = unidecode(text)

        print(f"[INFO] Đã xử lý: {os.path.basename(image_path)}")
        return {
            "Tên ảnh": os.path.basename(image_path),
            "Text gốc": text.strip(),
            "Text không dấu": text_no_diacritics.strip()
        }
    except Exception as e:
        print(f"Lỗi khi xử lý {image_path}: {e}")
        return {}

# Hàm xử lý toàn bộ thư mục ảnh
def process_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            result = process_image(image_path)
            if result:
                results.append(result)

    # Ghi kết quả ra Excel
    if results:
        df = pd.DataFrame(results)
        df.to_excel("thong_tin_cccd.xlsx", index=False)
        print("✅ Đã xuất kết quả ra thong_tin_cccd.xlsx")
    else:
        print("⚠️ Không có ảnh hợp lệ hoặc không nhận dạng được thông tin.")

# Gọi hàm chính
if __name__ == "__main__":
    process_folder(FOLDER_PATH)
