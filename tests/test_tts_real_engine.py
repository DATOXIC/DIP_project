import os
import struct
import threading
import time
import wave

import numpy as np
import pytest

import make_audio_cache
import speech_service
from tts_queue import QueuedSpeaker


@pytest.mark.skipif(os.name != "nt", reason="Cần SAPI trên Windows")
def test_real_sapi_reads_every_queued_sentence(monkeypatch):
    pytest.importorskip("win32com.client")
    monkeypatch.setenv("DIP_TTS_VOLUME", "0")
    speaker = QueuedSpeaker(rate=175, max_queue=4)
    elapsed = []
    original = speaker._speak_one

    def measured(text, voice_id):
        started = time.perf_counter()
        original(text, voice_id)
        elapsed.append(time.perf_counter() - started)

    speaker._speak_one = measured
    for text in ("Hello", "Thank you very much for coming", "Please"):
        assert speaker.speak(text)
        time.sleep(0.3)
    speaker.close(wait=True, timeout=10.0)
    assert len(elapsed) == 3
    assert all(duration >= 0.5 for duration in elapsed)


def test_queue_stats_keep_order_and_count_drops():
    spoken = []
    speaker = QueuedSpeaker(max_queue=8, speak_fn=lambda text: (spoken.append(text), time.sleep(0.01)))
    for text in ("Hello", "Thank you", "Please", "Sorry", "Yes"):
        assert speaker.speak(text)
    speaker.close()
    stats = speaker.stats()
    assert spoken == ["Hello", "Thank you", "Please", "Sorry", "Yes"]
    assert stats == {"enqueued": 5, "played": 5, "dropped": 0, "errors": 0}


def test_play_wav_runs_on_worker_thread(monkeypatch, tmp_path):
    called = []
    speaker = QueuedSpeaker()
    monkeypatch.setattr("winsound.PlaySound", lambda path, flags: called.append((path, flags, threading.current_thread())))
    wav_path = tmp_path / "silent.wav"
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 800)
    assert speaker.play_wav(str(wav_path))
    speaker.close()
    assert called and called[0][0] == str(wav_path)
    assert called[0][2] is not threading.main_thread()


def test_speech_service_uses_one_lazy_speaker(monkeypatch, tmp_path):
    fake = QueuedSpeaker(speak_fn=lambda text: None)
    monkeypatch.setattr(speech_service, "_speaker", fake)
    monkeypatch.setattr(speech_service, "_en_speaker", None)
    monkeypatch.setattr(speech_service, "AUDIO_CACHE_DIR", str(tmp_path))
    cache = tmp_path / "en"
    cache.mkdir()
    (cache / "HELLO.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(fake, "_play_wav", lambda path: None)
    assert speech_service.get_speaker() is fake
    assert speech_service.get_speaker() is fake
    assert speech_service.speak_word("HELLO", "en", "Hello") == "wav"
    assert speech_service.speak_word("MISSING", "en", "Missing") == "tts"
    fake.close()


def test_trim_synthetic_wav_removes_long_silence(tmp_path):
    sample_rate = 1000
    samples = np.zeros(2000, dtype="<i2")
    samples[110:1300] = 12000
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    measurement = make_audio_cache.trim_and_normalize_wav(str(source), str(target))
    assert measurement["leading"] <= 0.08
    assert measurement["trailing"] <= 0.12
    assert measurement["voiced"] >= 1.1
    make_audio_cache.validate_quality(measurement)
