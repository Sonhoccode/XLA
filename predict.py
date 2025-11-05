import cv2
from ultralytics import YOLO
import sys

# --- CẤU HÌNH ---

# 1. Trỏ đến file model "best.pt" bạn đã "thu hoạch"
MODEL_PATH = "best.pt"  # (Giả sử nó nằm chung thư mục với file predict.py)

# 2. CHỌN NGUỒN: '0' = webcam, 'video_test.mp4', 'anh_test.jpg'
SOURCE_TO_TEST = '0' 
# -----------------

def run_prediction():
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"LỖI: Không thể tải model từ '{MODEL_PATH}'.")
        print(f"Chi tiết lỗi: {e}")
        sys.exit()

    print(f"Bat dau du doan... Nhan 'q' de thoat.")
    
    try:
        # BƯỚC 1: Chạy model, nhưng TẮT tự động hiển thị (show=False)
        results = model(SOURCE_TO_TEST, 
                        stream=True, 
                        show=False) # <-- SỬA Ở ĐÂY

        # BƯỚC 2: Lặp qua từng kết quả
        for r in results:
            # BƯỚC 3: Tự vẽ kết quả lên ảnh VÀ CHỈ ĐỊNH FONT
            plotted_image = r.plot(font='Arial.Unicode.ttf',
                                   labels=True,  # Hiển thị tên (ví dụ: Cấm_vượt)
                                   conf=True)   # Hiển thị độ tự tin (ví dụ: 0.85)
            
            # BƯỚC 4: Tự hiển thị ảnh đã vẽ bằng OpenCV
            cv2.imshow("Ket Qua Nhan Dien (Nhan 'q' de thoat)", plotted_image)
            
            # Thoát bằng phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Loi khi du doan: {e}")

if __name__ == "__main__":
    run_prediction()