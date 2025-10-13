import cv2
from PIL import Image, ImageTk
import tkinter as tk

class LoadingOverlay(tk.Frame):
    def __init__(self, parent, avi_path, delay, fade_step=0.025):
		# === STEP 1: Prepare loader requirement ===
        # ------------------------------------
        
        #   1. Background
        super().__init__(parent, bg="white")
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        #   2. Video loader
        self.cap = cv2.VideoCapture(avi_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {avi_path}")

        #   3. Animation parameters
        self.delay = delay
        self.fade_step = fade_step
        self.label = tk.Label(self, bg="white")
        self.label.pack(expand=True)
        self.fading = False
        self.alpha = 1.0  # Full opacity

        #   4. Start animation
        self._animate()

    def _animate(self):
        # === STEP 1: Show frame ===
        # ------------------------------------
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # === STEP 2: Fade when triggered ===
            # ------------------------------------
            if self.fading:
                self.alpha -= self.fade_step
                self.alpha = max(self.alpha, 0.0)
                img = Image.blend(Image.new("RGB", img.size, (255, 255, 255)), img, self.alpha)
            tk_img = ImageTk.PhotoImage(img)
            self.label.config(image=tk_img)
            self.label.image = tk_img

		    # === STEP 3: Destroy loader after faded ===
            # ------------------------------------
            if self.fading and self.alpha <= 0:
                self.after(self.delay, self._destroy)
                return
            
		# === STEP 4: Resume animation ===
        # ------------------------------------
        self.after(self.delay, self._animate)

    def trigger_fade_and_destroy(self):
        # ====================================
		# === Helper to trigger fade from main script
        self.fading = True

    def _destroy(self):
        # ====================================
		# === Helper to destroy loader
        self.cap.release()
        self.destroy()
