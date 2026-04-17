"""Tests du lissage EMA des poses."""

from __future__ import annotations

import time

from marker_detection.pose_smoother import PoseSmoother, _angle_ema


def test_angle_ema_wrap_positive_to_negative():
    # 179 vers -179 : chemin le plus court = +2 degres, pas -358.
    out = _angle_ema(179.0, -179.0, 1.0)
    assert abs(((out - (-179.0) + 180) % 360) - 180) < 1e-6


def test_angle_ema_no_alpha_keeps_prev():
    assert _angle_ema(10.0, 20.0, 0.0) == 10.0


def test_smoother_converges_toward_target():
    sm = PoseSmoother(alpha_pos=0.5, alpha_angle=0.5, reset_timeout_s=10.0)
    # Premiere detection : initialise a la valeur mesuree.
    out1 = sm.smooth([("NUT_BLUE", 100.0, 50.0, 0.0)])
    assert out1[0][1:] == (100.0, 50.0, 0.0)
    # Deuxieme detection plus loin : EMA converge.
    out2 = sm.smooth([("NUT_BLUE", 200.0, 150.0, 90.0)])
    _, x, y, a = out2[0]
    assert 140.0 < x < 160.0
    assert 90.0 < y < 110.0
    assert 40.0 < a < 50.0


def test_smoother_reset_after_timeout():
    sm = PoseSmoother(alpha_pos=0.5, alpha_angle=0.5, reset_timeout_s=0.01)
    sm.smooth([("YR1", 0.0, 0.0, 0.0)])
    time.sleep(0.02)
    out = sm.smooth([("YR1", 100.0, 100.0, 90.0)])
    # Reset : retourne la valeur brute, pas une moyenne.
    assert out[0][1:] == (100.0, 100.0, 90.0)


def test_smoother_handles_duplicate_labels():
    sm = PoseSmoother(alpha_pos=1.0, alpha_angle=1.0, reset_timeout_s=10.0)
    out = sm.smooth([
        ("NUT_BLUE", 10.0, 20.0, 0.0),
        ("NUT_BLUE", 30.0, 40.0, 0.0),
    ])
    assert len(out) == 2
    assert out[0][1:3] == (10.0, 20.0)
    assert out[1][1:3] == (30.0, 40.0)
