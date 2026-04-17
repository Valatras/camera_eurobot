"""Serveur MJPEG leger pour streamer les frames annotees vers le dashboard."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOUNDARY = b"--frame\r\n"


class _MjpegHandler(BaseHTTPRequestHandler):
    """Handler HTTP qui sert un flux MJPEG sur GET /stream."""

    server: "_MjpegHTTPServer"

    def do_GET(self) -> None:
        if self.path != "/stream":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                jpeg = self.server.wait_frame()
                if jpeg is None:
                    continue
                self.wfile.write(BOUNDARY)
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        # Silence per-request logs
        pass


class _MjpegHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer avec stockage thread-safe de la derniere frame."""

    def __init__(self, port: int) -> None:
        super().__init__(("", port), _MjpegHandler)
        self._frame: bytes | None = None
        self._condition = threading.Condition()

    def set_frame(self, jpeg: bytes) -> None:
        with self._condition:
            self._frame = jpeg
            self._condition.notify_all()

    def wait_frame(self, timeout: float = 1.0) -> bytes | None:
        """Block until a new frame is available (or timeout)."""
        with self._condition:
            self._condition.wait(timeout=timeout)
            return self._frame


class MjpegServer:
    """Lance un serveur MJPEG dans un thread daemon."""

    def __init__(self, port: int = 8081) -> None:
        self._server = _MjpegHTTPServer(port)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_frame(self, jpeg: bytes) -> None:
        self._server.set_frame(jpeg)

    def stop(self) -> None:
        self._server.shutdown()
