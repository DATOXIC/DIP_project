"""Dựng cache WAV TTS đã cắt khoảng lặng cho các từ ASL."""

import argparse
import os
import shutil
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np



def trim_and_normalize_wav(input_path: str, output_path: str) -> dict:
    """Cắt khoảng lặng, làm mờ đuôi và chuẩn hóa WAV mono 16-bit."""
    with wave.open(input_path, "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError("WAV phải là mono 16-bit")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    if samples.size == 0:
        raise ValueError("WAV rỗng")
    peak = float(np.max(np.abs(samples)))
    if peak <= 0:
        raise ValueError("WAV không có tín hiệu")
    active = np.abs(samples) >= peak * 0.01
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        raise ValueError("WAV không có phần có tiếng")
    pad_start = int(round(sample_rate * 0.030))
    pad_end = int(round(sample_rate * 0.060))
    start = max(0, int(active_indices[0]) - pad_start)
    end = min(samples.size, int(active_indices[-1]) + pad_end + 1)
    trimmed = samples[start:end]
    fade_samples = min(trimmed.size, int(round(sample_rate * 0.015)))
    if fade_samples > 1:
        trimmed[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    normalized = trimmed * (26000.0 / max(1.0, float(np.max(np.abs(trimmed)))))
    normalized = np.clip(normalized, -32767, 32767).astype("<i2")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(output_path, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(normalized.tobytes())
    result = _measure_samples(normalized.astype(np.float32), sample_rate)
    result["peak"] = int(np.max(np.abs(normalized)))
    return result



def _measure_samples(samples: np.ndarray, sample_rate: int) -> dict:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    active = np.flatnonzero(np.abs(samples) >= peak * 0.01) if peak else np.array([], dtype=int)
    if active.size == 0:
        voiced = 0.0
        leading = trailing = samples.size / sample_rate
    else:
        voiced = (int(active[-1]) - int(active[0]) + 1) / sample_rate
        leading = int(active[0]) / sample_rate
        trailing = (samples.size - 1 - int(active[-1])) / sample_rate
    return {
        "total": samples.size / sample_rate,
        "voiced": voiced,
        "leading": leading,
        "trailing": trailing,
    }



def measure_wav(path: str) -> dict:
    with wave.open(path, "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("WAV phải là mono 16-bit")
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32)
        result = _measure_samples(samples, source.getframerate())
        result["peak"] = int(np.max(np.abs(samples))) if samples.size else 0
        return result



def validate_quality(measurement: dict) -> None:
    checks = (
        0.25 <= measurement["total"] <= 1.4,
        measurement["voiced"] >= 0.15,
        measurement["peak"] >= 3000,
        measurement["trailing"] <= 0.12,
        measurement["leading"] <= 0.08,
    )
    if not all(checks):
        raise ValueError(
            "Không đạt cổng chất lượng: "
            f"tổng={measurement['total']:.3f}, có_tiếng={measurement['voiced']:.3f}, "
            f"đỉnh={measurement['peak']}, đầu={measurement['leading']:.3f}, cuối={measurement['trailing']:.3f}"
        )



def _render_with_pyttsx3(text: str, rate: int, voice_id: str | None, output_path: str) -> None:
    """Dựng một file bằng engine riêng trong một luồng COM STA."""
    import pythoncom
    import pyttsx3

    pythoncom.CoInitialize()
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        engine.stop()
        if hasattr(pyttsx3, "_activeEngines"):
            pyttsx3._activeEngines.clear()
    finally:
        pythoncom.CoUninitialize()



def _render_one(text: str, rate: int, voice_id: str | None, output_path: str) -> None:
    error = []

    def worker() -> None:
        try:
            _render_with_pyttsx3(text, rate, voice_id, output_path)
        except Exception as exc:
            error.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    if not os.path.isfile(output_path):
        raise RuntimeError("pyttsx3 không tạo được file WAV")



def _word_keys() -> list[str]:
    import json

    with open("webcam_templates.json", "r", encoding="utf-8") as source:
        data = json.load(source)
    vocabulary = data.get("vocabulary", {})
    if isinstance(vocabulary, dict):
        return list(vocabulary.keys())
    return [str(item["word_key"]) for item in vocabulary]



def _phrase(word_key: str, lang: str) -> str:
    from predict_sentence_webcam import get_asl_phrase

    return get_asl_phrase(word_key, lang=lang, default=word_key)



def main() -> int:
    parser = argparse.ArgumentParser(description="Dựng cache WAV TTS đã cắt khoảng lặng")
    parser.add_argument("--lang", default="en", choices=("en", "vi"))
    parser.add_argument("--rate", type=int, default=175)
    parser.add_argument("--out", default="audio_cache")
    parser.add_argument("--voice-id", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.lang == "vi" and not args.voice_id:
        raise SystemExit("Cần --voice-id khi dựng cache tiếng Việt")
    if args.lang == "vi":
        raise SystemExit("Không dựng audio_cache/vi/ trên máy chưa có giọng Việt")

    keys = _word_keys()
    if len(keys) != 13:
        raise SystemExit(f"Cần đúng 13 word_key, thực tế có {len(keys)}")
    output_dir = Path(args.out) / args.lang
    temporary_dir = Path(tempfile.mkdtemp(prefix="dip_tts_"))
    try:
        for key in keys:
            destination = output_dir / f"{key}.wav"
            if args.check:
                if not destination.is_file():
                    raise SystemExit(f"Thiếu file {destination}")
                measurement = measure_wav(str(destination))
            else:
                raw_path = temporary_dir / f"{key}.wav"
                staged_path = temporary_dir / f"{key}_trimmed.wav"
                phrase = _phrase(key, args.lang)
                _render_one(phrase, args.rate, args.voice_id, str(raw_path))
                measurement = trim_and_normalize_wav(str(raw_path), str(staged_path))
                validate_quality(measurement)
                output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged_path, destination)
            print(
                f"{key}\t{_phrase(key, args.lang)}\t"
                f"{measurement['total']:.3f}s\tđỉnh={measurement['peak']}\t"
                f"có_tiếng={measurement['voiced']:.3f}s\t"
                f"đầu={measurement['leading']:.3f}s\tcuối={measurement['trailing']:.3f}s"
            )
        print(f"PASS: {len(keys)} file audio_cache/{args.lang}/")
        return 0
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
