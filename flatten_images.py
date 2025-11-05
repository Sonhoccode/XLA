import os
import shutil
from tqdm import tqdm

# Thư mục gốc chứa các thư mục con (0, 1, 2...)
# ĐÂY LÀ NƠI BẠN ĐÃ COPY ẢNH VÀO
SOURCE_IMAGE_DIR = r"C:\Users\H Son\Downloads\xulianh\dataset\images\train"

print(f"Bat dau lam phang thu muc: {SOURCE_IMAGE_DIR}")
print("Di chuyen tat ca file anh ra thu muc goc...")

# Lấy danh sách tất cả các mục trong thư mục nguồn
try:
    sub_dirs = [d for d in os.listdir(SOURCE_IMAGE_DIR) 
                if os.path.isdir(os.path.join(SOURCE_IMAGE_DIR, d))]
except FileNotFoundError:
    print(f"LOI: Khong tim thay thu muc {SOURCE_IMAGE_DIR}")
    exit()

if not sub_dirs:
    print("Thu muc da phang roi, khong can lam gi them.")
    exit()

total_files_moved = 0

# Sử dụng tqdm để xem tiến trình
for dir_name in tqdm(sub_dirs, desc="Dang xu ly cac thu muc con"):
    sub_dir_path = os.path.join(SOURCE_IMAGE_DIR, dir_name)
    
    try:
        files = [f for f in os.listdir(sub_dir_path) 
                 if os.path.isfile(os.path.join(sub_dir_path, f))]
    except Exception as e:
        print(f"Khong the doc thu muc {sub_dir_path}: {e}")
        continue
        
    for file_name in files:
        src_path = os.path.join(sub_dir_path, file_name)
        dest_path = os.path.join(SOURCE_IMAGE_DIR, file_name)
        
        # Di chuyển file
        shutil.move(src_path, dest_path)
        total_files_moved += 1
    
    # Sau khi di chuyển xong, xóa thư mục con rỗng
    try:
        os.rmdir(sub_dir_path)
    except OSError:
        print(f"Khong the xoa thu muc {sub_dir_path} (co the van con file)")

print(f"\n--- HOAN TAT ---")
print(f"Da di chuyen thanh cong {total_files_moved} file anh.")
print("Thu muc 'images/train' cua ban da duoc lam phang.")
print("Bay gio ban co the bat dau train!")