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

    def send_markers(self, detected: list[tuple[str, int, int]]) -> bool:
        """Envoie une liste de marqueurs a l'ESP32.

        Args:
            detected: Liste de tuples (label, grid_x, grid_y).

        Returns:
            True si toutes les donnees ont ete envoyees, False en cas d'erreur.
        """
        if not self.is_connected:
            logger.warning(
                "Non connecte — impossible d'envoyer les marqueurs.")
            return False

        try:
            for label, gx, gy in detected:
                line = f"{label},{gx},{gy}\n"
                # type: ignore[union-attr]
                self._conn.write(line.encode("ascii"))

            # End-of-frame marker
            self._conn.write(b"END\n")  # type: ignore[union-attr]
            self._conn.flush()  # type: ignore[union-attr]

            logger.debug("Envoye %d marqueur(s) a l'ESP32.", len(detected))
            return True

        except serial.SerialException as exc:
            logger.error("Erreur d'envoi serie : %s", exc)
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
