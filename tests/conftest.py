"""
Pytest Configuration & Shared Fixtures
======================================
Provides fixtures for DTW engines, segmenters, synthetic oracles, and simulators.
"""

import os
import sys
import pytest

# Ensure DIP_project directory is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dynamic_dtw_engine import (
    KineticState,
    KineticPauseSegmenter,
    HybridDTWEngine,
    HybridDTWConfig
)
from predict_sentence_webcam import (
    ASL_VOCAB_MAP,
    NonBlockingTTS,
    WristCoordinateSmoother
)
from tests.test_harness_utils import (
    SyntheticGestureOracle,
    HeadlessPipelineSimulator,
    TimerResolutionGuard
)


@pytest.fixture(scope="session")
def template_path():
    p = os.path.join(PROJECT_ROOT, "dynamic_templates.json")
    assert os.path.exists(p), f"Template database missing: {p}"
    return p


@pytest.fixture(scope="session")
def oracle(template_path):
    return SyntheticGestureOracle(templates_path=template_path)


@pytest.fixture
def dtw_engine(template_path):
    return HybridDTWEngine(templates_path=template_path)


@pytest.fixture
def segmenter():
    return KineticPauseSegmenter(
        v_start_ratio=0.08,
        v_stop_ratio=0.035,
        pause_frames=11,
        min_stroke_frames=8,
        max_stroke_frames=60,
        swipe_threshold=-0.80,
        swipe_ratio=2.5,
        swipe_cooldown_sec=1.0
    )


@pytest.fixture
def smoother():
    return WristCoordinateSmoother(alpha=0.70)


@pytest.fixture
def simulator(template_path):
    return HeadlessPipelineSimulator(templates_path=template_path)


@pytest.fixture
def tts():
    return NonBlockingTTS(speech_rate=175)


@pytest.fixture
def timer_guard():
    return TimerResolutionGuard()
