import threading
import time

import pytest

import speech_service
from tts_queue import QueuedSpeaker


def wait_for_thread(speaker, timeout=2.0):
    thread = speaker._thread
    if thread is not None:
        thread.join(timeout)
    assert thread is None or not thread.is_alive()


def test_two_words_are_read_in_order():
    spoken = []
    speaker = QueuedSpeaker(max_queue=4, speak_fn=lambda text: (spoken.append(text), time.sleep(0.02)))

    assert speaker.speak("Hello") is True
    assert speaker.speak("You") is True
    speaker.close()

    assert spoken == ["Hello", "You"]


def test_full_queue_discards_oldest_and_keeps_newest_order():
    spoken = []
    started = threading.Event()

    def fake_speak(text):
        spoken.append(text)
        if text == "one":
            started.set()
            time.sleep(0.05)

    speaker = QueuedSpeaker(max_queue=2, speak_fn=fake_speak)
    assert speaker.speak("one")
    assert started.wait(1.0)
    assert speaker.speak("two")
    assert speaker.speak("three")
    assert speaker.speak("four")
    speaker.close()

    assert spoken[0] == "one"
    assert spoken[1:] == ["three", "four"]
    assert "two" not in spoken
    assert len(spoken) <= 4


@pytest.mark.parametrize("value", [None, "", "   ", "\t\n"])
def test_invalid_text_is_rejected(value):
    spoken = []
    speaker = QueuedSpeaker(speak_fn=spoken.append)

    assert speaker.speak(value) is False
    speaker.close()
    assert spoken == []


def test_close_drains_queue_and_rejects_new_items():
    spoken = []
    speaker = QueuedSpeaker(speak_fn=lambda text: (spoken.append(text), time.sleep(0.01)))
    assert speaker.speak("one")
    assert speaker.speak("two")
    speaker.close(wait=True, timeout=1.0)

    assert spoken == ["one", "two"]
    assert speaker.speak("three") is False


def test_speak_returns_quickly_when_reader_is_slow():
    speaker = QueuedSpeaker(speak_fn=lambda text: time.sleep(0.25))
    started = time.perf_counter()
    assert speaker.speak("slow")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    speaker.close(timeout=1.0)

    assert elapsed_ms < 20.0


def test_reader_error_does_not_stop_worker():
    spoken = []

    def failing_reader(text):
        spoken.append(text)
        if text == "bad":
            raise RuntimeError("lỗi giả")

    speaker = QueuedSpeaker(speak_fn=failing_reader)
    assert speaker.speak("bad")
    assert speaker.speak("good")
    speaker.close()

    assert spoken == ["bad", "good"]


def test_wav_and_text_share_one_ordered_queue(monkeypatch):
    events = []
    speaker = QueuedSpeaker(speak_fn=lambda text: events.append(("text", text)))
    monkeypatch.setattr(speaker, "_play_wav", lambda path: events.append(("wav", path)))

    assert speaker.speak("Hello")
    assert speaker.play_wav("first.wav")
    assert speaker.speak("You")
    speaker.close()

    assert events == [("text", "Hello"), ("wav", "first.wav"), ("text", "You")]


def test_speech_service_muted_and_english_queue(monkeypatch):
    accepted = []
    monkeypatch.setattr(speech_service, "_en_speaker", lambda text: (accepted.append(text), True)[1])
    speech_service.set_muted(True)
    try:
        assert speech_service.speak_word("HELLO", "en", "Hello") == "muted"
        assert accepted == []
        speech_service.set_muted(False)
        assert speech_service.speak_word("HELLO", "en", "Hello") == "tts"
        assert accepted == ["Hello"]
    finally:
        speech_service.set_muted(False)
        speech_service.set_en_speaker(None)


def test_speech_service_vi_and_wav_use_queues(monkeypatch, tmp_path):
    vi_events = []
    wav_events = []
    vi_speaker = QueuedSpeaker(speak_fn=vi_events.append)
    wav_speaker = QueuedSpeaker(speak_fn=lambda path: wav_events.append(path))
    wav_speaker._play_wav = lambda path: wav_events.append(path)
    monkeypatch.setattr(speech_service, "_vi_speaker", vi_speaker)
    monkeypatch.setattr(speech_service, "_wav_speaker", wav_speaker)
    monkeypatch.setattr(speech_service, "_vi_checked", True)
    monkeypatch.setattr(speech_service, "_vi_voice_id", "fake-vi")
    wav_dir = tmp_path / "en"
    wav_dir.mkdir()
    wav_path = wav_dir / "HELLO.wav"
    wav_path.write_bytes(b"RIFF")
    monkeypatch.setattr(speech_service, "AUDIO_CACHE_DIR", str(tmp_path))

    try:
        assert speech_service.speak_word("HELLO", "vi", "Xin chao") == "tts"
        assert speech_service.speak_word("HELLO", "en", "Hello") == "wav"
        vi_speaker.close()
        wav_speaker.close()
        assert vi_events == ["Xin chao"]
        assert wav_events == [str(wav_path)]
    finally:
        vi_speaker.close(wait=False)
        wav_speaker.close(wait=False)
        speech_service.set_muted(False)
