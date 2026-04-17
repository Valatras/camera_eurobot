"""Gestion de la connexion serie USB vers un ESP32."""

from __future__ import annotations

import logging
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ESP32Sender:
    """Envoie les donnees de marqueurs a un ESP32 via USB/serie.

    Protocol:
        Chaque marqueur detecte est envoye sur une ligne :
            TYPE,X,Y\\n
        Suivi d'une ligne de fin de trame :
            END\\n

    Example on the ESP32 side (Arduino):
        void loop() {
            if (Serial.available()) {
                String line = Serial.readStringUntil('\\n');
                line.trim();
                if (line == "END") {
                    // process the frame
                } else {
                    // parse "TYPE,X,Y"
                }
            }
        }
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 1.0,
        auto_detect: bool = True,
    ) -> None:
        """Initialise le sender.

        Args:
            port: Port serie (ex: '/dev/ttyUSB0', 'COM3').
                  Si None et auto_detect=True, cherche automatiquement un ESP32.
            baudrate: Vitesse de communication (doit matcher l'ESP32).
            timeout: Timeout de lecture en secondes.
            auto_detect: Tente de detecter l'ESP32 automatiquement si port=None.
        """
        if not _SERIAL_AVAILABLE:
            raise ImportError(
                "Le module 'pyserial' est requis. "
                "Installez-le avec : pip install pyserial"
            )

        self._baudrate = baudrate
        self._timeout = timeout
        self._conn: Optional[serial.Serial] = None

        if port is None and auto_detect:
            port = self._find_esp32_port()

        self._port = port

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Ouvre la connexion serie.

        Returns:
            True si la connexion est etablie, False sinon.
        """
        if self._port is None:
            logger.error("Aucun port serie specifie ou detecte.")
            return False

        try:
            self._conn = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=self._timeout,
            )
            logger.info("Connecte a l'ESP32 sur %s @ %d baud.",
                        self._port, self._baudrate)
            return True
        except serial.SerialException as exc:
            logger.error("Impossible d'ouvrir %s : %s", self._port, exc)
            self._conn = None
            return False

    def disconnect(self) -> None:
        """Ferme la connexion serie proprement."""
        if self._conn and self._conn.is_open:
            self._conn.close()
            logger.info("Connexion serie fermee.")
        self._conn = None

    @property
    def is_connected(self) -> bool:
        """True si la connexion serie est ouverte."""
        return self._conn is not None and self._conn.is_open

    def __enter__(self) -> "ESP32Sender":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Data sending
    # ------------------------------------------------------------------

    def send_markers(self, detected: list[tuple[str, float, float, float]]) -> bool:
        """Envoie les positions des adversaires a l'ESP32.

        Seuls les marqueurs adverses sont envoyes, au format ``Obstacle X Y\\n``
        compatible avec le parser Commands.cpp du firmware robot.

        Args:
            detected: Liste de tuples (label, x_cm, y_cm, angle_deg).

        Returns:
            True si toutes les donnees ont ete envoyees, False en cas d'erreur.
        """
        if not self.is_connected:
            return False

        from marker_detection import config

        try:
            count = 0
            for label, x_cm, y_cm, _angle in detected:
                # Seuls les robots adverses sont envoyes.
                if config.TEAM_COLOR == "blue" and label.startswith("YR"):
                    pass  # opponent
                elif config.TEAM_COLOR == "yellow" and label.startswith("BR"):
                    pass  # opponent
                else:
                    continue

                line = f"Obstacle {int(round(x_cm))} {int(round(y_cm))}\n"
                self._conn.write(line.encode("ascii"))  # type: ignore[union-attr]
                count += 1

            # End-of-frame marker
            self._conn.write(b"END\n")  # type: ignore[union-attr]
            self._conn.flush()  # type: ignore[union-attr]

            logger.debug("Envoye %d obstacle(s) a l'ESP32.", count)
            return True

        except serial.SerialException as exc:
            logger.error("Erreur d'envoi serie : %s", exc)
            return False

    def send_vision_state(
        self,
        detected: list[tuple[str, float, float, float]],
        zone_sequences: dict[int, list[str]],
        gm_counts: dict[int, int] | None = None,
        stale: bool = False,
        corners_ok: bool = True,
    ) -> bool:
        """Envoie l'etat complet de la vision (protocole etendu).

        Ajoute de nouvelles lignes en plus du legacy ``Obstacle X Y``, de
        maniere retrocompatible : les anciens firmwares ignorent les lignes
        inconnues et continuent de parser ``Obstacle``. Le nouveau firmware
        (bridge etendu) consomme aussi ``OPP``, ``ZR``, ``GM`` et ``STALE``.

        Format textuel, une ligne par entite :

            STALE 1                      (si homographie obsolete)
            CORNERS_OK 0|1
            OPP x_cm y_cm theta_dcdeg    (un adversaire par ligne, angle*10)
            ZR id C1 C2 C3 C4 C5         (sequence couleur, 5 slots)
            GM id count                  (noisettes par garde-manger)
            Obstacle x y                 (legacy, meme info que OPP)
            END
        """
        if not self.is_connected:
            return False

        from marker_detection import config

        try:
            out: list[str] = []
            out.append(f"STALE {1 if stale else 0}\n")
            out.append(f"CORNERS_OK {1 if corners_ok else 0}\n")

            for label, x_cm, y_cm, angle in detected:
                is_opp = (
                    (config.TEAM_COLOR == "blue" and label.startswith("YR"))
                    or (config.TEAM_COLOR == "yellow" and label.startswith("BR"))
                )
                if not is_opp:
                    continue
                # Clamp to int16 domain + physical table bounds to guarantee
                # the firmware parser never sees an out-of-range value.
                xi = max(-32767, min(32767, int(round(x_cm))))
                yi = max(-32767, min(32767, int(round(y_cm))))
                ti = max(-1800, min(1800, int(round(angle * 10))))
                out.append(f"OPP {xi} {yi} {ti}\n")
                # Legacy ligne pour retrocompat.
                out.append(f"Obstacle {xi} {yi}\n")

            for zid, seq in zone_sequences.items():
                # Warn if we somehow got more than 5 slots — symptom of a
                # classification bug upstream.
                if len(seq) > 5:
                    logger.warning(
                        "ZR %d: %d slots (expected ≤5), truncating", zid, len(seq))
                    seq = seq[:5]
                seq_str = " ".join(seq)
                out.append(f"ZR {int(zid)} {seq_str}\n")

            if gm_counts:
                for gid, count in gm_counts.items():
                    c = max(0, min(127, int(count)))
                    out.append(f"GM {int(gid)} {c}\n")

            out.append("END\n")

            payload = "".join(out).encode("ascii")
            self._conn.write(payload)  # type: ignore[union-attr]
            self._conn.flush()  # type: ignore[union-attr]
            return True

        except serial.SerialException as exc:
            logger.error("Erreur d'envoi serie (vision_state) : %s", exc)
            return False

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_esp32_port() -> Optional[str]:
        """Tente de detecter automatiquement le port USB de l'ESP32.

        Cherche les USB VID/PID connus des puces ESP32 (CP210x, CH340, FTDI).

        Returns:
            Le nom du port detecte, ou None si aucun trouve.
        """
        # (VID, PID) couples courants pour les puces USB des ESP32
        KNOWN_VID_PID = {
            (0x10C4, 0xEA60),  # Silicon Labs CP210x
            (0x1A86, 0x7523),  # CH340
            (0x0403, 0x6001),  # FTDI FT232R
            (0x0403, 0x6015),  # FTDI FT231X
            (0x303A, 0x1001),  # Espressif USB JTAG/serial (ESP32-S3/C3 natif)
        }

        for port_info in serial.tools.list_ports.comports():
            if (port_info.vid, port_info.pid) in KNOWN_VID_PID:
                logger.info(
                    "ESP32 detecte automatiquement sur %s (%s).",
                    port_info.device,
                    port_info.description,
                )
                return port_info.device

        logger.warning(
            "Aucun ESP32 detecte automatiquement. "
            "Specifiez le port manuellement (ex: port='/dev/ttyUSB0')."
        )
        return None

    @staticmethod
    def list_available_ports() -> list[str]:
        """Liste tous les ports serie disponibles (utile pour le debug).

        Returns:
            Liste des noms de ports detectes sur le systeme.
        """
        ports = [p.device for p in serial.tools.list_ports.comports()]
        logger.info("Ports serie disponibles : %s", ports or "aucun")
        return ports
