"""Tests for multi-signal batch analysis (pipeline_batch.py)."""

import numpy as np
import pytest

from prototype.pipeline_batch import analyze_all_candidates, candidate_timeline


def _two_tone_capture(seed=5):
    """Strong QPSK-like tone pair region + weaker pure carrier, well separated."""
    rng = np.random.default_rng(seed)
    fs = 8000.0
    t = np.arange(16000) / fs

    strong = 0.9 * np.exp(1j * 2 * np.pi * 500.0 * t)
    weak = 0.3 * np.exp(1j * 2 * np.pi * 2800.0 * t)
    noise = 0.01 * (
        rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size)
    )
    return strong + weak + noise, fs


class TestBatchAnalysis:
    def test_detects_multiple_candidates(self):
        x, fs = _two_tone_capture()
        batch = analyze_all_candidates(x, fs, mode="quick")
        assert len(batch.detections) >= 2

    def test_results_ranked_by_strength(self):
        x, fs = _two_tone_capture()
        batch = analyze_all_candidates(x, fs, mode="quick")
        assert batch.results, "expected at least one analyzed candidate"
        first_index = batch.results[0][0]
        # strongest candidate is analyzed first
        assert first_index == 0

    def test_max_candidates_limits_work(self):
        x, fs = _two_tone_capture()
        batch = analyze_all_candidates(x, fs, mode="quick", max_candidates=1)
        assert len(batch.results) <= 1

    def test_second_candidate_is_the_weak_tone(self):
        from dataclasses import replace

        from prototype.core.config import processing_mode_config

        x, fs = _two_tone_capture()
        config = replace(processing_mode_config("quick"), candidate_index=1)
        batch = analyze_all_candidates(x, fs, config=config, mode="quick")
        analyzed = [r for i, r in batch.results if i == 1]
        assert analyzed
        center = analyzed[0].selected_candidate["center_frequency"]
        assert abs(center - 2800.0) < 300.0

    def test_timeline_sorted_by_frequency(self):
        x, fs = _two_tone_capture()
        batch = analyze_all_candidates(x, fs, mode="quick")
        timeline = candidate_timeline(batch, fs)
        freqs = [row["center_frequency_hz"] for row in timeline]
        assert freqs == sorted(freqs)
        assert all({"modulation", "classification_confidence"} <= set(row) for row in timeline)

    def test_to_dict_serializable(self):
        x, fs = _two_tone_capture()
        batch = analyze_all_candidates(x, fs, mode="quick")
        import json

        payload = json.dumps(batch.to_dict())
        assert "analyzed_candidates" in payload

    def test_no_candidates_graceful(self):
        rng = np.random.default_rng(1)
        noise = 0.5 * (rng.standard_normal(8192) + 1j * rng.standard_normal(8192))
        batch = analyze_all_candidates(noise, 8000.0, mode="quick")
        assert batch.detections == []
        assert batch.results == []
        assert batch.warnings
