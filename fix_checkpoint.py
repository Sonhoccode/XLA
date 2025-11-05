# File: fix_checkpoint.py
import torch

OLD_MODEL_PATH = 'last.pt'
NEW_MODEL_PATH = 'last_FIXED.pt' # Chúng ta sẽ tạo ra file mới

print(f"Dang tai file checkpoint: {OLD_MODEL_PATH}")
try:
    # Tải checkpoint lên CPU
    ckpt = torch.load(OLD_MODEL_PATH, map_location='cpu')

    # Lấy phần 'args' (trí nhớ) của nó ra
    # Kiểm tra xem 'args' có tồn tại không
    if 'args' in ckpt:
        args = ckpt['args']
        print("Tim thay 'args' trong checkpoint. Bat dau xoa cac duong dan cu...")
        
        # Xóa các đường dẫn "cứng" của Colab
        if hasattr(args, 'project'):
            print(f" - Xoa project cu: {args.project}")
            delattr(args, 'project')
            
        if hasattr(args, 'name'):
            print(f" - Xoa name cu: {args.name}")
            delattr(args, 'name')
            
        if hasattr(args, 'save_dir'):
            print(f" - Xoa save_dir cu: {args.save_dir}")
            delattr(args, 'save_dir')
        
        # Cập nhật lại args vào checkpoint
        ckpt['args'] = args
        
        # Lưu checkpoint đã sửa
        torch.save(ckpt, NEW_MODEL_PATH)
        print(f"\n--- THANH CONG! ---")
        print(f"Da tao file moi ten: {NEW_MODEL_PATH}")
        print("File nay da 'quen' duong dan C:\\content\\... cu.")

    else:
        print("LOI: Khong tim thay 'args' trong file checkpoint. File co the bi hong.")

except Exception as e:
    print(f"LOI KHI DOC/LUU FILE: {e}")