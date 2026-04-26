import cv2
from PIL import Image, ImageTk
import tkinter as tk


class LoadingOverlay(tk.Frame):
    def __init__(self, parent, avi_path, delay, fade_step=0.025):
        super().__init__(parent, bg="white")
        self.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.cap = cv2.VideoCapture(avi_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {avi_path}")

        self.delay = delay
        self.fade_step = fade_step
        self.label = tk.Label(self, bg="white")
        self.label.pack(expand=True)
        self.fading = False
        self.alpha = 1.0
        self._destroyed = False

        self._animate()

    def _animate(self):
        if self._destroyed:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            if self.fading:
                self.alpha -= self.fade_step
                self.alpha = max(self.alpha, 0.0)
                img = Image.blend(Image.new("RGB", img.size, (255, 255, 255)), img, self.alpha)

            tk_img = ImageTk.PhotoImage(img)
            self.label.config(image=tk_img)
            self.label.image = tk_img

            if self.fading and self.alpha <= 0:
                self.after(self.delay, self._destroy)
                return

        self.after(self.delay, self._animate)

    def trigger_fade_and_destroy(self):
        try:
            self.after(0, self._begin_fade)
        except tk.TclError:
            pass

    def _begin_fade(self):
        if not self._destroyed:
            self.fading = True

    def _destroy(self):
        if self._destroyed:
            return

        self._destroyed = True

        try:
            self.cap.release()
        except Exception:
            pass

        try:
            self.label.configure(image="")
            self.label.image = None
        except Exception:
            pass

        try:
            self.destroy()
        except tk.TclError:
            pass