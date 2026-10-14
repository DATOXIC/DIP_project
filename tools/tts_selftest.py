"""Tự kiểm tra hàng đợi TTS bằng số hoặc phát thử có tiếng."""

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import speech_service



def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / float(source.getframerate())



def _run_silent(keys: list[str]) -> int:
    speaker = speech_service.get_speaker()
    events = []
    cache_dir = Path(speech_service.AUDIO_CACHE_DIR) / "en"
    durations = {key: _wav_duration(cache_dir / f"{key}.wav") for key in keys if (cache_dir / f"{key}.wav").is_file()}
    average_duration = sum(durations.values()) / len(durations) if durations else 0.35

    def silent_wav(path: str) -> None:
        started = time.perf_counter()
        time.sleep(durations.get(Path(path).stem, average_duration))
        events.append((Path(path).stem, started, time.perf_counter()))

    original_play_wav = speaker._play_wav
    speaker._play_wav = silent_wav
    try:
        queued_at = time.perf_counter()
        for key in keys:
            if speech_service.speak_word(key, "en", key) not in ("wav", "tts"):
                print(f"FAIL: không xếp được {key}")
                return 1
            time.sleep(0.3)
        speaker.close(wait=True, timeout=10.0)
    finally:
        speaker._play_wav = original_play_wav
        speech_service.shutdown()

    stats = speaker.stats()
    expected_order = keys[: len(events)]
    durations_played = [end - start for _, start, end in events]
    order_ok = [key for key, _, _ in events] == expected_order
    no_overlap = all(events[index + 1][1] >= events[index][2] - 0.02 for index in range(len(events) - 1))
    all_duration_ok = all(duration >= 0.25 for duration in durations_played)
    last_start_limit = 5.0 * average_duration + 1.0
    last_start_ok = bool(events) and events[-1][1] - queued_at <= last_start_limit
    pass_all = (
        stats["played"] == stats["enqueued"] - stats["dropped"]
        and stats["dropped"] == 0
        and len(events) == len(keys)
        and order_ok
        and no_overlap
        and all_duration_ok
        and last_start_ok
    )
    print(f"stats={stats}")
    print(f"thời_lượng_trung_bình={sum(durations_played) / len(durations_played):.3f}s")
    for key, started, ended in events:
        print(f"{key}\tbắt_đầu={started - queued_at:.3f}s\tkết_thúc={ended - queued_at:.3f}s\tdài={ended - started:.3f}s")
    print("PASS" if pass_all else "FAIL")
    return 0 if pass_all else 1



def _run_listen(keys: list[str]) -> int:
    for key in keys:
        speech_service.speak_word(key, "en", key)
        time.sleep(0.3)
    speaker = speech_service.get_speaker()
    speaker.close(wait=True, timeout=20.0)
    speech_service.shutdown()
    print("Bạn nghe được mấy từ, theo thứ tự nào?")
    return 0



def main() -> int:
    parser = argparse.ArgumentParser(description="Tự kiểm tra đường đọc âm thanh")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--silent", action="store_true")
    mode.add_argument("--listen", action="store_true")
    args = parser.parse_args()
    keys = ["HELLO", "THANK_YOU", "PLEASE", "SORRY", "YES"]
    speech_service.set_muted(False)
    speech_service.set_en_speaker(None)
    return _run_silent(keys) if args.silent else _run_listen(keys)


if __name__ == "__main__":
    raise SystemExit(main())
