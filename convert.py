import pandas as pd
import os
from tqdm import tqdm

# --- Cấu hình ---
TRAIN_CSV_PATH = "Train.csv"  # Đường dẫn tới file Train.csv của bạn
TEST_CSV_PATH = "Test.csv"    # Đường dẫn tới file Test.csv của bạn

OUTPUT_DIR = "dataset"
TRAIN_IMG_DIR = os.path.join(OUTPUT_DIR, "images/train")
VAL_IMG_DIR = os.path.join(OUTPUT_DIR, "images/val")
TRAIN_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels/train")
VAL_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels/val")

# Tạo các thư mục output nếu chưa có
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)

# --- Hàm tính toán ---
def convert_to_yolo_format(row):
    """
    Chuyển đổi từ tọa độ (x1, y1, x2, y2) sang (x_center, y_center, width, height)
    và chuẩn hóa (normalize) về 0-1.
    """
    img_width = row['Width']
    img_height = row['Height']
    
    x1 = row['Roi.X1']
    y1 = row['Roi.Y1']
    x2 = row['Roi.X2']
    y2 = row['Roi.Y2']
    
    class_id = row['ClassId']
    
    # Tính toán width và height của bounding box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Tính toán tọa độ tâm (center)
    x_center = x1 + (box_width / 2)
    y_center = y1 + (box_height / 2)
    
    # Chuẩn hóa (Normalize)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = box_width / img_width
    height_norm = box_height / img_height
    
    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"

def process_csv(csv_path, label_dir):
    """
    Đọc file CSV, xử lý từng dòng và ghi ra file .txt tương ứng.
    """
    print(f"Đang xử lý file: {csv_path}...")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {csv_path}. Hãy đảm bảo bạn để đúng đường dẫn.")
        return

    # Sử dụng tqdm để xem tiến trình
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        # Lấy tên file ảnh gốc từ cột 'Path' 
        # Ví dụ: "Train/0/00000_00000_00000.png"
        original_path = row['Path']
        
        # Chỉ lấy tên file, bỏ qua phần "Train/..."
        # Kết quả: "00000_00000_00000.png"
        base_filename = os.path.basename(original_path)
        
        # Đổi đuôi .png -> .txt
        # Kết quả: "00000_00000_00000.txt"
        txt_filename = os.path.splitext(base_filename)[0] + ".txt"
        
        # Đường dẫn file .txt output
        output_txt_path = os.path.join(label_dir, txt_filename)
        
        # Tính toán tọa độ YOLO
        yolo_line = convert_to_yolo_format(row)
        
        # Ghi vào file (chế độ 'a' - append để thêm nếu ảnh có nhiều biển báo)
        with open(output_txt_path, 'a') as f:
            f.write(yolo_line)

# --- Chạy thực thi ---
print("Bắt đầu chuyển đổi dữ liệu Train...")
process_csv(TRAIN_CSV_PATH, TRAIN_LABEL_DIR)

print("\nBắt đầu chuyển đổi dữ liệu Test (Validation)...")
process_csv(TEST_CSV_PATH, VAL_LABEL_DIR)

print("\nHoàn tất! Dữ liệu của bạn đã sẵn sàng trong thư mục 'dataset'.")