import sys, runpy
sys.path.insert(0, r"C:\DIP\DIP_project")
variant, video, csvp = sys.argv[1:4]
import pipeline_common as pc

class Pass:
    def update(self, x, y, timestamp=None): return float(x), float(y)
    def reset(self): pass

class Ema:
    def __init__(self, a): self.a, self.p = a, None
    def update(self, x, y, timestamp=None):
        self.p = (float(x), float(y)) if self.p is None else (self.a * x + (1 - self.a) * self.p[0], self.a * y + (1 - self.a) * self.p[1])
        return self.p
    def reset(self): self.p = None

extra = []
if variant == "no1e":
    pc.make_wrist_smoother = lambda *a, **k: Pass()
elif variant == "ema65":
    pc.make_wrist_smoother = lambda *a, **k: Ema(0.65)
elif variant == "nolk":
    pc.LKInertialFallback.predict_wrist = lambda self, g: None
elif variant.endswith("_enh"):
    extra = ["--dip-enhance"]
sys.argv = ["predict_sentence_webcam.py", "--video", video, "--video-clock", "--headless", "--mute", "--one-shot",
            "--spotter", "dl", "--handshape-gate", "--no-dl", "--log-session", csvp] + extra
runpy.run_path(r"C:\DIP\DIP_project\predict_sentence_webcam.py", run_name="__main__")
