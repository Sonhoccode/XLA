from ultralytics import YOLO

# --- CẤU HÌNH ---
CHECKPOINT_PATH = 'last.pt' 
CONFIG_PATH = 'data.yaml' 
# --- KẾT THÚC CẤU HÌNH ---

def main():
    print("!!! PHIEN BAN CHUAN 100% DA SUA LOI SAVE !!!") 
    model = YOLO(CHECKPOINT_PATH) 
    print(f"Bat dau huan luyen tiep tu checkpoint: {CHECKPOINT_PATH}")
    print(f"Su dung file config local: {CONFIG_PATH}")
    
    try:
        results = model.train(
            # Ghi đè file config của Colab
            data=CONFIG_PATH,     
            
            # Báo cho model huấn luyện tiếp
            resume=True,          
            
            # BẮT BUỘC: Ghi đè đường dẫn LƯU FILE của Colab
            project='.',  # Lưu vào thư mục 'runs' tại thư mục hiện tại
            name='yolov8_local_resumed_final', # Đặt tên mới cho lần chạy local này
            
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