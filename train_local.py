# File: train_local.py (Sửa lại)
from ultralytics import YOLO

# 1. Tên file model TỐT NHẤT bạn vừa tải về
CHECKPOINT_PATH = 'best.pt' # <-- SỬA CHỖ NÀY

# 2. Tên file config của máy local
CONFIG_PATH = 'data.yaml' 

def main():
    print(f"!!! DANG CHAY FILE MOI: train_local.py !!!") 
    model = YOLO(CHECKPOINT_PATH) 

    print(f"Bat dau huan luyen tiep tu checkpoint TỐT NHẤT: {CHECKPOINT_PATH}")
    print(f"Su dung file config local: {CONFIG_PATH}")
    
    try:
        results = model.train(
            # Ghi đè file config
            data=CONFIG_PATH,     
            
            # Báo cho model huấn luyện tiếp
            resume=True,          
            
            # Ghi đè đường dẫn lưu file
            project='.',  # Lưu vào thư mục 'runs' tại thư mục hiện tại
            name='yolov8_local_resumed_from_best', # Tên thư mục MỚI
            
            # Các tham số khác
            epochs=300,
            patience=50,
            imgsz=640,
            batch=16,
            workers=8
        )
        
        print("---  HOAN THANH TRAINING! ---")
        
    except Exception as e:
        print(f"Gap loi: {e}")

if __name__ == '__main__':
    main()