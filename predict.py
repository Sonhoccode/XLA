import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading

# --- CẤU HÌNH ---
MODEL_PATH = "best.pt"
FONT_PATH = 'Arial.Unicode.ttf'
# -----------------

# Tải model (chỉ tải 1 lần)
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Lỗi Model", f"Không thể tải model: {e}")
    exit()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Trình Nhận Diện Biển Báo Giao Thông (YOLOv8)")
        self.root.geometry("800x600")

        # Biến kiểm soát camera
        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None

        # --- Tạo các Tab ---
        self.tab_control = ttk.Notebook(root)
        
        self.tab_image = ttk.Frame(self.tab_control)
        self.tab_webcam = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab_image, text='Nhận Diện Ảnh')
        self.tab_control.add(self.tab_webcam, text='Nhận Diện Webcam')
        
        self.tab_control.pack(expand=1, fill="both")

        # --- Thiết kế Tab 1: Nhận Diện Ảnh ---
        self.setup_image_tab()
        
        # --- Thiết kế Tab 2: Nhận Diện Webcam ---
        self.setup_webcam_tab()
        
        # Đảm bảo tắt camera khi đóng cửa sổ
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_image_tab(self):
        # Nút bấm
        btn_load = ttk.Button(self.tab_image, text="Chọn Ảnh", command=self.load_image)
        btn_load.pack(pady=20)
        
        # Khung hiển thị ảnh
        self.image_label = ttk.Label(self.tab_image)
        self.image_label.pack(padx=10, pady=10, fill="both", expand=True)

    def setup_webcam_tab(self):
        # Khung hiển thị video
        self.webcam_label = ttk.Label(self.tab_webcam)
        self.webcam_label.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Nút bấm
        self.btn_start_cam = ttk.Button(self.tab_webcam, text="Bật Camera", command=self.start_webcam)
        self.btn_start_cam.pack(side="left", padx=50, pady=10)
        
        self.btn_stop_cam = ttk.Button(self.tab_webcam, text="Tắt Camera", command=self.stop_webcam, state="disabled")
        self.btn_stop_cam.pack(side="right", padx=50, pady=10)

    def predict_and_show(self, image_bgr):
        # Chạy model
        results = model(image_bgr)
        r = results[0]
        
        # Vẽ kết quả (với font tiếng Việt)
        annotated_image_bgr = r.plot(font=FONT_PATH, labels=True, conf=True)
        
        # Chuyển BGR (OpenCV) -> RGB (Tkinter)
        image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        
        # Chuyển thành ảnh của Tkinter
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        return img_tk

    def load_image(self):
        # Mở hộp thoại chọn file
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if not file_path:
            return
            
        # Đọc ảnh bằng OpenCV
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            messagebox.showerror("Lỗi Ảnh", "Không thể đọc file ảnh.")
            return

        # Nhận diện và hiển thị
        img_tk = self.predict_and_show(image_bgr)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk # Giữ tham chiếu

    def start_webcam(self):
        if self.is_camera_running:
            return
            
        try:
            self.cap = cv2.VideoCapture(0) # Thử camera 0
            if not self.cap.isOpened():
                 self.cap = cv2.VideoCapture(1) # Thử camera 1
            if not self.cap.isOpened():
                raise IOError("Không thể mở webcam (0 hoặc 1).")
        except Exception as e:
            messagebox.showerror("Lỗi Webcam", str(e))
            return

        self.is_camera_running = True
        self.btn_start_cam.config(state="disabled")
        self.btn_stop_cam.config(state="normal")
        
        # Tạo luồng (thread) mới để chạy camera, tránh làm "đơ" UI
        self.camera_thread = threading.Thread(target=self.webcam_loop)
        self.camera_thread.daemon = True # Tự tắt thread khi đóng app
        self.camera_thread.start()

    def webcam_loop(self):
        while self.is_camera_running:
            ret, frame_bgr = self.cap.read()
            if not ret:
                print("Lỗi: Không thể lấy khung hình webcam.")
                break
                
            # Nhận diện (giống hệt hàm load_image)
            img_tk = self.predict_and_show(frame_bgr)
            
            # Cập nhật UI
            self.webcam_label.config(image=img_tk)
            self.webcam_label.image = img_tk
        
        # Dọn dẹp
        if self.cap:
            self.cap.release()
        self.btn_start_cam.config(state="normal")
        self.btn_stop_cam.config(state="disabled")

    def stop_webcam(self):
        self.is_camera_running = False # Tín hiệu để thread dừng lại
        print("Đang dừng camera...")

    def on_closing(self):
        self.stop_webcam() # Dừng camera
        self.root.destroy() # Đóng cửa sổ

# --- Chạy App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()