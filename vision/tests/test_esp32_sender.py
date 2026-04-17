"""Tests du protocole texte emis par ESP32Sender.

On stub la connexion serie avec un objet ``io.BytesIO``-like qui
capture tous les bytes ecrits, pour valider le format des lignes
sans dependre de ``pyserial``.
"""

from __future__ import annotations

from marker_detection.esp32_sender import ESP32Sender


class _FakeSerial:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.is_open = True

    def write(self, data: bytes) -> int:
        self.buffer.extend(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


def _make_sender() -> tuple[ESP32Sender, _FakeSerial]:
    try:
        s = ESP32Sender(port="/dev/null", auto_detect=False)
    except ImportError:
        import pytest
        pytest.skip("pyserial non installe")
    fake = _FakeSerial()
    s._conn = fake  # type: ignore[assignment]  # noqa: SLF001
    return s, fake


def test_send_vision_state_emits_opp_with_clamped_values():
    from marker_detection import config
    s, fake = _make_sender()
    config.TEAM_COLOR = "blue"
    # Adversaire = YR si on est bleu.
    s.send_vision_state(
        detected=[("YR1", 50000.0, -50000.0, 720.0)],  # hors bornes
        zone_sequences={},
        stale=False,
        corners_ok=True,
    )
    text = fake.buffer.decode("ascii")
    # Les valeurs doivent etre clampees.
    assert "OPP 32767 -32767 1800" in text
    # La ligne legacy doit aussi etre presente.
    assert "Obstacle 32767 -32767" in text
    assert text.endswith("END\n")


def test_send_vision_state_emits_zr_and_gm():
    from marker_detection import config
    s, fake = _make_sender()
    config.TEAM_COLOR = "blue"
    s.send_vision_state(
        detected=[],
        zone_sequences={3: ["B", "Y", "E", "?", "?"]},
        gm_counts={1: 2, 5: 200},  # 200 doit etre clampe a 127
        stale=True,
        corners_ok=False,
    )
    text = fake.buffer.decode("ascii")
    assert "STALE 1" in text
    assert "CORNERS_OK 0" in text
    assert "ZR 3 B Y E ? ?" in text
    assert "GM 1 2" in text
    assert "GM 5 127" in text


def test_send_vision_state_ignores_ally_as_opponent():
    from marker_detection import config
    s, fake = _make_sender()
    config.TEAM_COLOR = "blue"
    s.send_vision_state(
        detected=[("BR1", 100.0, 100.0, 0.0)],  # ally, pas adversaire
        zone_sequences={},
    )
    text = fake.buffer.decode("ascii")
    assert "OPP" not in text
    assert "Obstacle" not in text
