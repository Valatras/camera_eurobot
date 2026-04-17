#!/usr/bin/env python3
"""Relay camera: capture V4L2 → sert un flux MJPEG brut sur HTTP.

Tourne sur le laptop (pas de detection, juste capture + JPEG encode).
Le serveur distant consomme ce flux via CAMERA_URL.

Usage:
    python camera_relay.py [--device 2] [--port 8082] [--width 3840] [--height 2160]
    nix run .#camera-relay -- --device 2 --port 8082
"""

from __future__ import annotations

import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = b"--frame\r\n"


class _Handler(BaseHTTPRequestHandler):
    server: "_RelayServer"

    def do_GET(self) -> None:
        if self.path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
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
                self.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass


class _RelayServer(ThreadingHTTPServer):
    def __init__(self, port: int) -> None:
        super().__init__(("", port), _Handler)
        self._frame: bytes | None = None
        self._cond = threading.Condition()

    def set_frame(self, jpeg: bytes) -> None:
        with self._cond:
            self._frame = jpeg
            self._cond.notify_all()

    def wait_frame(self, timeout: float = 1.0) -> bytes | None:
        with self._cond:
            self._cond.wait(timeout=timeout)
            return self._frame


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Camera relay (V4L2 → MJPEG HTTP)"
    )
    ap.add_argument(
        "--device", type=int, default=2, help="V4L2 device index (default: 2)"
    )
    ap.add_argument(
        "--port", type=int, default=8082, help="HTTP listen port (default: 8082)"
    )
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality 1-100 (default: 85)",
    )
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open /dev/video{args.device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    srv = _RelayServer(args.port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"\U0001f4f7 Camera relay: /dev/video{args.device} "
        f"({actual_w}x{actual_h}) → http://0.0.0.0:{args.port}/stream"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            _, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
            )
            srv.set_frame(jpeg.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        srv.shutdown()


if __name__ == "__main__":
    main()
