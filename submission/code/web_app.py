# -*- coding: utf-8 -*-
"""
BẢN WEB: nhận diện từ VIDEO, chọn mốc bắt đầu/kết thúc ngay trên trình phát.

Chạy:   py dip.py web            (hoặc  py web_app.py)   ->  http://127.0.0.1:8000

Cách hoạt động (không viết lại thuật toán nào):
    trình duyệt  --(tải file / liên kết / video mẫu)-->  server lưu video vào web_uploads/
    trình duyệt  --(mốc bắt đầu, kết thúc, hồ sơ)--->    server chạy CHÍNH predict_sentence_webcam.py ở chế độ
                 headless với  --video --start --end --events  (cùng bộ phân đoạn, DTW và tầng DL dự phòng như bản webcam)
    server đọc nhật ký sự kiện JSONL do pipeline ghi (mỗi cử chỉ: từ, mốc thời gian VIDEO, nguồn DTW/DL, top-3 ứng viên)
    và trả về để trang vẽ lên dòng thời gian.

An toàn: chỉ lắng nghe 127.0.0.1, từ chối Host/Origin không phải localhost, chỉ nhận đuôi video đã liệt kê, giới hạn
dung lượng/thời lượng, tham số truyền cho pipeline được ép kiểu và kiểm tra (không nhận đường dẫn/cờ tùy ý từ trình duyệt).
"""
from __future__ import annotations

import argparse
import ast
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest, urlopen

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import people

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "web_uploads"
RUN_DIR = UPLOAD_DIR / "_runs"
SAMPLES_DIR = ROOT / "ASL_Citizen_videos"
INDEX_HTML = ROOT / "web" / "index.html"

ALLOWED_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
CONTENT_TYPE_EXT = {"video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
                    "video/x-matroska": ".mkv", "video/x-msvideo": ".avi", "video/x-m4v": ".m4v"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_DURATION_SEC = 600.0
MAX_JOB_SEC = 900.0
KEEP_UPLOAD_SEC = 24 * 3600
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

@asynccontextmanager
async def lifespan(_app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_uploads()
    _restore_uploads()
    yield


app = FastAPI(title="DIP - Nhận diện từ video", lifespan=lifespan)

MEDIA: Dict[str, Dict[str, Any]] = {}
JOBS: Dict[str, Dict[str, Any]] = {}
_state_lock = threading.Lock()
_active_job: Optional[str] = None


# ============================================================================ tiện ích

def _load_vocab_names() -> Dict[str, Dict[str, str]]:
    """Đọc ASL_VOCAB_MAP thẳng từ mã nguồn pipeline (không import: tránh nạp torch/mediapipe) -> một nguồn sự thật."""
    src = (ROOT / "predict_sentence_webcam.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        if target == "ASL_VOCAB_MAP" and node.value is not None:
            return ast.literal_eval(node.value)
    return {}


VOCAB_NAMES = _load_vocab_names()


def _names(word: Optional[str]) -> Dict[str, str]:
    info = VOCAB_NAMES.get(word or "", {})
    return {"en": info.get("en", word or ""), "vi": info.get("vi", "")}


def _probe(path: Path) -> Dict[str, Any]:
    """Mở video bằng OpenCV để lấy thông số; ném 400 nếu không đọc được hoặc quá dài."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise HTTPException(400, "Không đọc được file video này (định dạng/codec không được OpenCV hỗ trợ).")
        fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps_raw if (fps_raw == fps_raw and fps_raw > 1.0) else 30.0
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if frames <= 0:
        raise HTTPException(400, "Video không có khung hình đọc được.")
    duration = frames / fps
    if duration > MAX_DURATION_SEC:
        raise HTTPException(400, f"Video dài {duration:.0f}s, vượt giới hạn {MAX_DURATION_SEC:.0f}s. Hãy cắt ngắn trước.")
    return {"duration": round(duration, 3), "fps": round(fps, 3), "width": width, "height": height}


def _register(path: Path, display: str, source: str) -> Dict[str, Any]:
    info = _probe(path)
    media_id = path.stem if source != "sample" else uuid.uuid4().hex[:16]
    rec = {"id": media_id, "path": path, "name": display[:90], "source": source, **info}
    MEDIA[media_id] = rec
    return _public_media(rec)


def _public_media(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": rec["id"], "name": rec["name"], "source": rec["source"], "duration": rec["duration"],
            "fps": rec["fps"], "width": rec["width"], "height": rec["height"], "url": f"/media/{rec['id']}"}


def _safe_name(name: str) -> str:
    base = os.path.basename(name or "video")
    return "".join(ch for ch in base if ch.isprintable() and ch not in '<>:"/\\|?*')[:90] or "video"


def _cleanup_old_uploads() -> None:
    now = time.time()
    if not UPLOAD_DIR.exists():
        return
    for p in list(UPLOAD_DIR.iterdir()):
        try:
            if p.is_file() and now - p.stat().st_mtime > KEEP_UPLOAD_SEC:
                p.unlink()
        except OSError:
            pass
    if RUN_DIR.exists():
        for p in list(RUN_DIR.iterdir()):
            try:
                if p.is_file() and now - p.stat().st_mtime > KEEP_UPLOAD_SEC:
                    p.unlink()
            except OSError:
                pass


def _restore_uploads() -> None:
    """Sau khi khởi động lại server, các video đã tải lên (còn trong web_uploads/) vẫn dùng lại được."""
    if not UPLOAD_DIR.exists():
        return
    for p in UPLOAD_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            try:
                info = _probe(p)
            except HTTPException:
                continue
            MEDIA[p.stem] = {"id": p.stem, "path": p, "name": p.name, "source": "upload", **info}


# ============================================================================ bảo vệ chỉ-localhost

@app.middleware("http")
async def local_only(request: Request, call_next):
    host = urlparse("//" + (request.headers.get("host") or "")).hostname or ""
    if host.lower() not in LOCAL_HOSTS:
        return JSONResponse({"detail": "Chỉ truy cập được qua localhost."}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and (urlparse(origin).hostname or "").lower() not in LOCAL_HOSTS:
            return JSONResponse({"detail": "Nguồn (Origin) không hợp lệ."}, status_code=403)
    return await call_next(request)


# ============================================================================ trang, video, danh mục

@app.get("/")
def index() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(500, "Thiếu web/index.html")
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})


@app.get("/media/{media_id}")
def media(media_id: str) -> FileResponse:
    rec = MEDIA.get(media_id)
    if rec is None or not Path(rec["path"]).is_file():
        raise HTTPException(404, "Không tìm thấy video.")
    ctype = mimetypes.guess_type(str(rec["path"]))[0] or "video/mp4"
    return FileResponse(rec["path"], media_type=ctype)


@app.get("/api/vocab")
def api_vocab() -> List[Dict[str, str]]:
    return [{"word": w, "en": n["en"], "vi": n["vi"]} for w, n in _load_vocab_names().items()]


@app.get("/api/profiles")
def api_profiles() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pid in [people.OWNER_ID] + list(people.volunteer_ids()):
        st = people.profile_stats(pid)
        if st["mau"] <= 0:
            continue
        label = people.display_name(pid)
        if pid == people.OWNER_ID:
            label += " (bank sản xuất)"
        out.append({"id": pid, "name": label, "words": st["tu"], "samples": st["mau"]})
    return out


@app.get("/api/samples")
def api_samples() -> List[Dict[str, str]]:
    if not SAMPLES_DIR.is_dir():
        return []
    items: List[Dict[str, str]] = []
    for word_dir in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        for f in sorted(word_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
                items.append({"word": word_dir.name, "file": f.name, "path": f"{word_dir.name}/{f.name}"})
    return items


class SampleReq(BaseModel):
    path: str


@app.post("/api/sample")
def api_use_sample(req: SampleReq) -> Dict[str, Any]:
    target = (SAMPLES_DIR / req.path).resolve()
    if SAMPLES_DIR.resolve() not in target.parents or not target.is_file() or target.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, "Video mẫu không hợp lệ.")
    return _register(target, f"Mẫu: {req.path}", "sample")


# ============================================================================ nhập video

@app.post("/api/upload")
def api_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Chỉ nhận video: {', '.join(sorted(ALLOWED_EXT))}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    media_id = uuid.uuid4().hex[:16]
    dest = UPLOAD_DIR / f"{media_id}{ext}"
    size = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = file.file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File vượt giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
                out.write(chunk)
        return _register(dest, _safe_name(file.filename or "video"), "upload")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


class FetchReq(BaseModel):
    url: str


@app.post("/api/fetch")
def api_fetch(req: FetchReq) -> Dict[str, Any]:
    parsed = urlparse(req.url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(400, "Chỉ nhận liên kết http/https trỏ thẳng tới file video.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    request = UrlRequest(req.url.strip(), headers={"User-Agent": "Mozilla/5.0 (DIP video fetch)"})
    try:
        resp = urlopen(request, timeout=20)
    except Exception as exc:
        raise HTTPException(400, f"Không tải được liên kết: {exc}")
    with resp:
        ctype = (resp.headers.get_content_type() or "").lower()
        ext = Path(parsed.path).suffix.lower()
        if ext not in ALLOWED_EXT:
            ext = CONTENT_TYPE_EXT.get(ctype, "")
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, "Liên kết này không phải file video trực tiếp (trang YouTube/TikTok... chưa được hỗ trợ; "
                                     "hãy tải video về rồi dùng mục 'Tải file').")
        declared = int(resp.headers.get("Content-Length") or 0)
        if declared > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File vượt giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        media_id = uuid.uuid4().hex[:16]
        dest = UPLOAD_DIR / f"{media_id}{ext}"
        size = 0
        try:
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"File vượt giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
                    out.write(chunk)
            return _register(dest, _safe_name(Path(parsed.path).name or parsed.hostname), "url")
        except BaseException:
            dest.unlink(missing_ok=True)
            raise


# ============================================================================ chạy pipeline

# Cách cắt cử chỉ -> cờ của pipeline. "force" = cả đoạn [start, end] là MỘT cử chỉ (lấy mẫu), không dùng bộ phân đoạn.
SEGMENTERS: Dict[str, List[str]] = {
    "standard": [],
    "fast": ["--no-ready-pose"],
    "fast_hold": ["--no-ready-pose", "--hold-on-hand-loss"],
    "force": ["--force-stroke"],
}
# Cách dự đoán -> (dùng DL dự phòng, cờ thêm). "plain" = tắt chuỗi nhanh (SPEC 08) và cắt dẫn đầu (SPEC 08B).
PREDICTORS: Dict[str, Any] = {
    "dtw_dl": (True, []),
    "dtw": (False, []),
    "dtw_dl_plain": (True, ["--no-fast-chain", "--no-lead-trim"]),
    "dtw_plain": (False, ["--no-fast-chain", "--no-lead-trim"]),
}


# Tinh chỉnh bộ phân đoạn (khoảng hợp lệ). Giá trị được ép kiểu và kẹp kiểm tra trước khi thành cờ dòng lệnh.
TUNE_LIMITS: Dict[str, Any] = {
    "pause_sec": ("--pause-sec", 0.03, 1.5),     # tay khựng bao lâu thì vào chờ chốt
    "grace_sec": ("--grace-sec", 0.0, 1.0),      # chờ thêm trước khi chốt hẳn
    "v_stop": ("--v-stop", 0.005, 0.2),          # ngưỡng đứng yên
    "v_start": ("--v-start", 0.02, 0.5),         # ngưỡng bắt đầu ký
}


def _tune_flags(tune: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    for key, value in (tune or {}).items():
        if key == "motion":
            if value not in ("wrist", "hand"):
                raise HTTPException(400, "Tinh chỉnh 'motion' chỉ nhận 'wrist' hoặc 'hand'.")
            flags += ["--motion", str(value)]
            continue
        if key == "early_commit":
            if not isinstance(value, bool):
                raise HTTPException(400, "Tinh chỉnh 'early_commit' phải là đúng/sai.")
            flags.append("--early-commit" if value else "--no-early-commit")
            continue
        if key not in TUNE_LIMITS:
            raise HTTPException(400, f"Tham số tinh chỉnh không hợp lệ: {key}")
        flag, lo, hi = TUNE_LIMITS[key]
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise HTTPException(400, f"Giá trị của '{key}' phải là số.")
        if not (lo <= num <= hi):
            raise HTTPException(400, f"'{key}' phải nằm trong khoảng {lo} đến {hi}.")
        flags += [flag, f"{num:.4f}"]
    return flags


class AnalyzeReq(BaseModel):
    media_id: str
    start: float = 0.0
    end: Optional[float] = None
    profile: str = people.OWNER_ID
    preroll: float = 1.0
    segmenter: str = "fast"      # cách CẮT cử chỉ (khi thu): xem SEGMENTERS
    predictor: str = "dtw_dl"    # cách DỰ ĐOÁN: xem PREDICTORS
    tune: Dict[str, Any] = {}    # tinh chỉnh bộ phân đoạn: xem TUNE_LIMITS (khóa vắng = giữ mặc định)


def _read_events(path: Path, start: float, keep_sample: bool = False) -> List[Dict[str, Any]]:
    """Đọc JSONL sự kiện. `idx` = thứ tự dòng trong file (ổn định khi sắp xếp/lọc), dùng để chỉ đúng sự kiện khi lưu mẫu.
    Mẫu thô (`sample`, 32x24 số) chỉ giữ lại khi keep_sample=True; danh sách trả cho trình duyệt chỉ có cờ has_sample."""
    events: List[Dict[str, Any]] = []
    if not path.is_file():
        return events
    idx = -1
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):
                break
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            idx += 1
            if float(ev.get("t_end", 0.0)) < start - 0.05:
                continue
            ev["idx"] = idx
            sample = ev.pop("sample", None)
            ev["has_sample"] = sample is not None
            if keep_sample:
                ev["sample"] = sample
            ev["name"] = _names(ev.get("word"))
            for key in ("dtw_word", "dl_word"):
                ev[key + "_name"] = _names(ev.get(key)) if ev.get(key) else None
            for c in ev.get("candidates", []):
                c["name"] = _names(c.get("word"))
            events.append(ev)
    events.sort(key=lambda e: e.get("t_start", 0.0))
    return events


def _run_job(job_id: str) -> None:
    global _active_job
    job = JOBS[job_id]
    cmd = [sys.executable, "predict_sentence_webcam.py", "--video", str(job["video_path"]), "--headless", "--mute",
           "--no-confirm-enroll", "--events", str(job["events_path"]),
           "--start", f"{job['seek']:.3f}", "--templates", job["bank"]]
    if job["end"] is not None:
        cmd += ["--end", f"{job['end']:.3f}"]
    cmd += ["--dl"] if job["use_dl"] else ["--no-dl"]
    cmd += SEGMENTERS[job["segmenter"]] + PREDICTORS[job["predictor"]][1] + job["tune_flags"]
    cmd += ["--trace", str(job["trace_path"])]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    job["status"] = "running"
    job["t0"] = time.time()
    proc: Optional[subprocess.Popen] = None
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", creationflags=flags)
        job["proc"] = proc
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if line.startswith("@@PROGRESS"):
                try:
                    job["progress"] = float(line.split()[1])
                except (IndexError, ValueError):
                    pass
                continue
            if line:
                job["log"].append(line)
                if len(job["log"]) > 80:
                    del job["log"][:-80]
            if job["cancel"]:
                proc.terminate()
                break
            if time.time() - job["t0"] > MAX_JOB_SEC:
                job["error"] = f"Quá {MAX_JOB_SEC:.0f}s, đã dừng."
                proc.terminate()
                break
        code = proc.wait(timeout=30)
        if job["cancel"]:
            job["status"] = "cancelled"
        elif job["error"]:
            job["status"] = "error"
        elif code != 0:
            job["status"] = "error"
            job["error"] = f"Pipeline thoát với mã {code}. Xem nhật ký bên dưới."
        else:
            job["progress"] = 1.0
            job["status"] = "done"
    except Exception as exc:
        job["status"] = "error"
        job["error"] = f"Lỗi chạy pipeline: {exc}"
        if proc is not None and proc.poll() is None:
            proc.kill()
    finally:
        job["t1"] = time.time()
        job["proc"] = None
        with _state_lock:
            if _active_job == job_id:
                _active_job = None


@app.post("/api/analyze")
def api_analyze(req: AnalyzeReq) -> Dict[str, Any]:
    global _active_job
    rec = MEDIA.get(req.media_id)
    if rec is None:
        raise HTTPException(404, "Chưa chọn video hoặc video đã bị xóa.")
    duration = float(rec["duration"])
    if not (0.0 <= req.start < duration):
        raise HTTPException(400, f"Mốc bắt đầu phải nằm trong [0, {duration:.2f}) giây.")
    end = None if req.end is None else min(float(req.end), duration)
    if end is not None and end <= req.start + 0.3:
        raise HTTPException(400, "Mốc kết thúc phải lớn hơn mốc bắt đầu ít nhất 0.3 giây.")
    if req.segmenter not in SEGMENTERS:
        raise HTTPException(400, "Cách cắt cử chỉ không hợp lệ.")
    if req.predictor not in PREDICTORS:
        raise HTTPException(400, "Cách dự đoán không hợp lệ.")
    tune_flags = _tune_flags(req.tune)
    force = req.segmenter == "force"
    if force and end is None:
        raise HTTPException(400, "Chế độ lấy mẫu (ép đoạn) cần đặt cả mốc kết thúc.")
    preroll = 0.0 if force else min(max(float(req.preroll), 0.0), 3.0)
    profiles = {p["id"] for p in api_profiles()}
    if req.profile not in profiles:
        raise HTTPException(400, "Hồ sơ không hợp lệ hoặc chưa có dữ liệu.")
    bank = people.profile_path(req.profile)
    with _state_lock:
        if _active_job is not None:
            raise HTTPException(409, "Đang có một lần nhận diện chạy; hãy chờ hoặc hủy nó trước.")
        job_id = uuid.uuid4().hex[:12]
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        JOBS[job_id] = {
            "id": job_id, "media_id": req.media_id, "video_path": rec["path"], "start": float(req.start), "end": end,
            "seek": max(0.0, float(req.start) - preroll), "preroll": preroll, "bank": bank, "profile": req.profile,
            "use_dl": PREDICTORS[req.predictor][0], "segmenter": req.segmenter, "predictor": req.predictor, "force": force, "tune_flags": tune_flags,
            "events_path": RUN_DIR / f"{job_id}.jsonl", "trace_path": RUN_DIR / f"{job_id}.trace.jsonl", "status": "queued", "progress": 0.0,
            "log": [], "error": "", "cancel": False, "proc": None, "t0": time.time(), "t1": None,
        }
        _active_job = job_id
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Không có tác vụ này.")
    events = _read_events(Path(job["events_path"]), job["start"])
    t_end = job["t1"] if job["t1"] else time.time()
    accepted = [e for e in events if e["event"] == "ACCEPTED"]
    return {
        "id": job_id, "status": job["status"], "progress": round(float(job["progress"]), 4), "error": job["error"],
        "elapsed": round(t_end - job["t0"], 1), "start": job["start"], "end": job["end"], "seek": job["seek"],
        "profile": job["profile"], "use_dl": job["use_dl"], "force": job["force"], "segmenter": job["segmenter"], "predictor": job["predictor"], "events": events, "log": job["log"][-40:],
        "summary": {
            "accepted": len(accepted),
            "uncertain": sum(1 for e in events if e["event"] == "UNCERTAIN"),
            "rejected": sum(1 for e in events if e["event"] == "REJECTED"),
            "dl_decided": sum(1 for e in accepted if e.get("source") == "DL"),
            "sentence": [e["name"]["en"] for e in accepted],
        },
    }


@app.get("/api/jobs/{job_id}/trace")
def api_trace(job_id: str) -> Dict[str, Any]:
    """Vết từng khung đã xử lý: t (giây video), h (có tay), v (vận tốc/lòng bàn tay), s (trạng thái FSM), lm (21 mốc xy chuẩn hóa)."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Không có tác vụ này.")
    rows: List[Dict[str, Any]] = []
    path = Path(job["trace_path"])
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.endswith("\n"):
                    break
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if float(r.get("t", 0.0)) >= job["start"] - 1e-3:
                    rows.append(r)
    return {"frames": rows}


class SampleSaveReq(BaseModel):
    idx: int
    word: str       # từ thật do NGƯỜI xác nhận (không phải dự đoán của hệ thống)
    person: str     # tên người ký trong video (người lạ); tạo hồ sơ mới nếu chưa có


@app.post("/api/jobs/{job_id}/sample")
def api_save_sample(job_id: str, req: SampleSaveReq) -> Dict[str, Any]:
    """Lưu mẫu của một sự kiện vào hồ sơ của người ký: profiles/<mã>.json. MỘT file này dùng chung cho cả hai tầng:
    DTW (dùng làm bank với --ho-so) và DL (train_dynamic_dl.py quét profiles/*.json, mỗi file là một người ký).
    KHÔNG bao giờ ghi vào bank sản xuất (webcam_templates.json)."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Không có tác vụ này.")
    if job["status"] not in ("done", "cancelled", "error"):
        raise HTTPException(409, "Hãy chờ nhận diện chạy xong rồi mới lưu mẫu.")
    word = req.word.strip()
    if word not in _load_vocab_names():
        raise HTTPException(400, "Từ này không thuộc từ vựng của hệ thống.")
    name = req.person.strip()
    if not name or len(name) > 40:
        raise HTTPException(400, "Hãy nhập tên người ký (tối đa 40 ký tự).")
    existing = people.resolve(name)
    if existing == people.OWNER_ID:
        raise HTTPException(400, "Không ghi vào bank sản xuất của Đạt; hãy dùng tên người ký khác.")
    matches = [e for e in _read_events(Path(job["events_path"]), job["start"], keep_sample=True) if e["idx"] == req.idx]
    if not matches or not matches[0].get("sample"):
        raise HTTPException(404, "Sự kiện này không có mẫu để lưu.")
    ev = matches[0]
    try:
        pid = existing or people.add_person(name)   # chỉ tạo hồ sơ mới SAU khi mọi kiểm tra đầu vào đã qua
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    sample = dict(ev["sample"])
    sample.update({
        "chirality_canonical": False, "rep_count": None, "is_original": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "web_video", "web_media": job["media_id"], "web_t_start": ev["t_start"], "web_t_end": ev["t_end"],
        "web_forced": bool(ev.get("forced")),
    })
    path = Path(people.PROFILES_DIR) / f"{pid}.json"
    with _state_lock:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {"metadata": {"feature_dim": 24, "source": "web_video"}, "vocabulary": {}}
        entry = data.setdefault("vocabulary", {}).setdefault(word, {"samples": []})
        samples = entry.setdefault("samples", [])
        key = (sample["web_media"], sample["web_t_start"], sample["web_t_end"])
        samples[:] = [x for x in samples if (x.get("web_media"), x.get("web_t_start"), x.get("web_t_end")) != key]
        samples.append(sample)
        entry["num_samples"] = len(samples)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    return {"person_id": pid, "person_name": people.display_name(pid), "word": word,
            "word_samples": len(samples), "total_samples": sum(len(v.get("samples", [])) for v in data["vocabulary"].values()),
            "file": f"profiles/{pid}.json"}


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str) -> Dict[str, str]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Không có tác vụ này.")
    job["cancel"] = True
    proc = job.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
    return {"status": "cancelling"}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bản web nhận diện từ video (chỉ chạy trên localhost).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Không tự mở trình duyệt.")
    args = parser.parse_args(argv)
    import uvicorn
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nBản web nhận diện từ video: {url}  (Ctrl+C để dừng)\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
