import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import time

# --- CẤU HÌNH ---
MODEL_PATH = "best.pt"
FONT_PATH = 'Arial.Unicode.ttf'
# -----------------

print("Đang tải model YOLOv8... Vui lòng đợi.")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Lỗi Model", f"Không thể tải model: {e}")
    exit()
print("Tải model thành công!")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Trình Nhận Diện Biển Báo Giao Thông (YOLOv8)")
        self.root.geometry("900x800") 

        # Biến kiểm soát
        self.cap_cam = None # Cho webcam
        self.cap_vid = None # Cho video file
        self.is_camera_running = False
        self.is_video_running = False
        self.camera_thread = None
        self.video_thread = None

        # --- Tạo các Tab ---
        self.tab_control = ttk.Notebook(root, bootstyle="primary")
        
        self.tab_image = ttk.Frame(self.tab_control, padding=10)
        self.tab_webcam = ttk.Frame(self.tab_control, padding=10)
        self.tab_video = ttk.Frame(self.tab_control, padding=10) # <-- TAB MỚI
        
        self.tab_control.add(self.tab_image, text='  Nhận Diện Ảnh  ')
        self.tab_control.add(self.tab_webcam, text='  Nhận Diện Webcam  ')
        self.tab_control.add(self.tab_video, text='  Nhận Diện Video File  ') # <-- TAB MỚI
        
        self.tab_control.pack(expand=1, fill="both")

        # --- Thiết kế các Tab ---
        self.setup_image_tab()
        self.setup_webcam_tab()
        self.setup_video_tab() # <-- TAB MỚI
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- CÁC HÀM SETUP GIAO DIỆN ---
    
    def setup_image_tab(self):
        # Frame chính
        main_frame = ttk.Frame(self.tab_image)
        main_frame.pack(fill="both", expand=True)

        btn_load = ttk.Button(main_frame, text="Chọn Ảnh Từ Máy Tính", 
                             command=self.load_image, bootstyle="primary-outline", padding=10)
        btn_load.pack(pady=10)
        
        # Khung hiển thị ảnh
        self.image_label = ttk.Label(main_frame, text="Kết quả nhận diện ảnh sẽ hiển thị ở đây",
                                     bootstyle="secondary", relief="solid", anchor=CENTER)
        self.image_label.pack(padx=10, pady=10, fill="both", expand=True)

        # Khung kết quả text
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(results_frame, text="Kết quả chi tiết:", bootstyle="info").pack(side="left", padx=5)
        self.image_results_text = ttk.Text(results_frame, height=5, state="disabled", font=("Arial", 10))
        self.image_results_text.pack(fill="x", expand=True)

    def setup_webcam_tab(self):
        # Frame chính
        main_frame = ttk.Frame(self.tab_webcam)
        main_frame.pack(fill="both", expand=True)

        # Khung hiển thị video
        self.webcam_label = ttk.Label(main_frame, text="Camera sẽ hiển thị ở đây",
                                      bootstyle="secondary", relief="solid", anchor=CENTER)
        self.webcam_label.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Frame nút bấm
        cam_btn_frame = ttk.Frame(main_frame)
        cam_btn_frame.pack(fill="x", pady=10)
        self.btn_start_cam = ttk.Button(cam_btn_frame, text="Bật Camera", 
                                        command=self.start_webcam, bootstyle="success", padding=10)
        self.btn_start_cam.pack(side="left", padx=50, pady=10, expand=True)
        self.btn_stop_cam = ttk.Button(cam_btn_frame, text="Tắt Camera", 
                                       command=self.stop_webcam, bootstyle="danger-outline", state="disabled", padding=10)
        self.btn_stop_cam.pack(side="right", padx=50, pady=10, expand=True)

        # Khung kết quả text
        results_frame_cam = ttk.Frame(main_frame)
        results_frame_cam.pack(fill="x", pady=(10, 0))
        ttk.Label(results_frame_cam, text="Kết quả chi tiết:", bootstyle="info").pack(side="left", padx=5)
        self.webcam_results_text = ttk.Text(results_frame_cam, height=5, state="disabled", font=("Arial", 10))
        self.webcam_results_text.pack(fill="x", expand=True)

    def setup_video_tab(self):
        # Frame chính
        main_frame = ttk.Frame(self.tab_video)
        main_frame.pack(fill="both", expand=True)

        # Khung hiển thị video
        self.video_label = ttk.Label(main_frame, text="Video sẽ phát ở đây",
                                      bootstyle="secondary", relief="solid", anchor=CENTER)
        self.video_label.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Frame nút bấm
        vid_btn_frame = ttk.Frame(main_frame)
        vid_btn_frame.pack(fill="x", pady=10)
        self.btn_start_vid = ttk.Button(vid_btn_frame, text="Chọn File Video & Chạy", 
                                        command=self.start_video_processing, bootstyle="success", padding=10)
        self.btn_start_vid.pack(side="left", padx=50, pady=10, expand=True)
        self.btn_stop_vid = ttk.Button(vid_btn_frame, text="Dừng Video", 
                                       command=self.stop_video_processing, bootstyle="danger-outline", state="disabled", padding=10)
        self.btn_stop_vid.pack(side="right", padx=50, pady=10, expand=True)

        # Khung kết quả text
        results_frame_vid = ttk.Frame(main_frame)
        results_frame_vid.pack(fill="x", pady=(10, 0))
        ttk.Label(results_frame_vid, text="Kết quả chi tiết:", bootstyle="info").pack(side="left", padx=5)
        self.video_results_text = ttk.Text(results_frame_vid, height=5, state="disabled", font=("Arial", 10))
        self.video_results_text.pack(fill="x", expand=True)

    # --- CÁC HÀM XỬ LÝ ---

    def predict_and_show(self, image_bgr, target_image_label, target_text_widget):
        """Hàm nhận diện chung cho cả ảnh và video/webcam"""
        
        # 1. Chạy model
        results = model(image_bgr, verbose=False) # verbose=False để tắt log
        r = results[0]
        
        # 2. Cập nhật Frame kết quả (Text)
        results_string = f"Phát hiện {len(r.boxes)} đối tượng:\n"
        results_string += "---------------------------------\n"
        if len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label_name = model.names[cls_id]
                results_string += f"> {label_name} (Độ tự tin: {conf*100:.1f}%)\n"
        else:
            results_string = "Không phát hiện biển báo nào."

        target_text_widget.config(state="normal")
        target_text_widget.delete("1.0", tk.END)
        target_text_widget.insert("1.0", results_string)
        target_text_widget.config(state="disabled")

        
        # 3. Vẽ kết quả với line_width mỏng hơn (fix font scale)
        annotated_image_bgr = r.plot(font=FONT_PATH, labels=True, conf=True, line_width=2) 
        
        # 4. Lấy kích thước của khung chứa (container/label)
        container = target_image_label
        container.update_idletasks() 
        container_w = container.winfo_width() - 2 
        container_h = container.winfo_height() - 2
        if container_w < 50 or container_h < 50: container_w, container_h = 780, 540 
            
        # 5. Lấy kích thước ảnh kết quả và tính tỷ lệ (fix scale méo)
        img_h, img_w, _ = annotated_image_bgr.shape
        ratio = min(container_w / img_w, container_h / img_h)
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)

        # 6. Chuyển đổi và resize
        image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # 7. Cập nhật ảnh lên UI
        target_image_label.config(image=img_tk, text="")
        target_image_label.image = img_tk 

    def load_image(self):
        # Dừng các stream khác (nếu đang chạy)
        self.stop_webcam()
        self.stop_video_processing()
            
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if not file_path:
            return
            
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            messagebox.showerror("Lỗi Ảnh", "Không thể đọc file ảnh.")
            return
        
        self.predict_and_show(image_bgr, self.image_label, self.image_results_text)

    # --- Logic cho Webcam ---
    
    def start_webcam(self):
        # Dừng các stream khác
        self.stop_video_processing()
        if self.is_camera_running:
            return
        
        self.tab_control.select(self.tab_webcam) # Tự chuyển tab
            
        try:
            self.cap_cam = cv2.VideoCapture(0) 
            if not self.cap_cam.isOpened(): self.cap_cam = cv2.VideoCapture(1)
            if not self.cap_cam.isOpened():
                raise IOError("Không thể mở webcam (0 hoặc 1).")
        except Exception as e:
            messagebox.showerror("Lỗi Webcam", str(e))
            return

        self.is_camera_running = True
        self.btn_start_cam.config(state="disabled")
        self.btn_stop_cam.config(state="normal")
        
        self.camera_thread = threading.Thread(target=self.webcam_loop)
        self.camera_thread.daemon = True 
        self.camera_thread.start()

    def webcam_loop(self):
        try:
            while self.is_camera_running:
                ret, frame_bgr = self.cap_cam.read()
                if not ret:
                    print("Lỗi: Không thể lấy khung hình webcam.")
                    break
                
                self.root.after(0, self.predict_and_show, frame_bgr, self.webcam_label, self.webcam_results_text)
                time.sleep(0.01) # Giảm tải CPU
        except Exception as e:
            print(f"Lỗi trong webcam_loop: {e}")
        
        if self.cap_cam: self.cap_cam.release()
        self.btn_start_cam.config(state="normal")
        self.btn_stop_cam.config(state="disabled")

    def stop_webcam(self):
        self.is_camera_running = False
        print("Đang dừng webcam...")

    # --- (MỚI) Logic cho Video File ---

    def start_video_processing(self):
        # Dừng các stream khác
        self.stop_webcam()
        if self.is_video_running:
            self.stop_video_processing()
        
        self.tab_control.select(self.tab_video) # Tự chuyển tab
        
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        if not file_path:
            return
            
        try:
            self.cap_vid = cv2.VideoCapture(file_path)
            if not self.cap_vid.isOpened():
                raise IOError("Không thể mở file video đã chọn.")
        except Exception as e:
            messagebox.showerror("Lỗi Video", str(e))
            return

        self.is_video_running = True
        self.btn_start_vid.config(state="disabled")
        self.btn_stop_vid.config(state="normal")
        
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.daemon = True 
        self.video_thread.start()

    def video_loop(self):
        try:
            while self.is_video_running:
                ret, frame_bgr = self.cap_vid.read()
                if not ret:
                    print("Video đã kết thúc.")
                    break # Tự động dừng khi hết video
                
                self.root.after(0, self.predict_and_show, frame_bgr, self.video_label, self.video_results_text)
                time.sleep(0.01) # Giảm tải CPU, bạn có thể điều chỉnh
        except Exception as e:
            print(f"Lỗi trong video_loop: {e}")
        
        # Tự động dọn dẹp khi vòng lặp kết thúc (do hết video hoặc nhấn 'Dừng')
        self.is_video_running = False
        if self.cap_vid: self.cap_vid.release()
        self.btn_start_vid.config(state="normal")
        self.btn_stop_vid.config(state="disabled")
        self.video_results_text.config(state="normal")
        self.video_results_text.insert(tk.END, "\n--- VIDEO ĐÃ KẾT THÚC ---")
        self.video_results_text.config(state="disabled")


    def stop_video_processing(self):
        self.is_video_running = False
        print("Đang dừng video file...")

    # --- HÀM ĐÓNG APP ---
    
    def on_closing(self):
        print("Đang đóng ứng dụng...")
        self.stop_webcam() 
        self.stop_video_processing()
        self.root.destroy() 

# --- Chạy App ---
if __name__ == "__main__":
    # Chọn theme: "superhero" (tối), "litera" (sáng), "cyborg", "darkly"
    root = ttk.Window(themename="superhero") 
    app = App(root)
    root.mainloop()