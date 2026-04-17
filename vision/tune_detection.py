#!/usr/bin/env python3
"""Outil interactif pour tuner les parametres de detection ArUco.

Permet de tester rapidement différentes valeurs de CLAHE, blur, et seuils
avant de les committer dans config.py/runtime.py.

Usage:
  python3 tune_detection.py --help
  python3 tune_detection.py --image path/to/frame.jpg  # Test sur image statique
  python3 tune_detection.py --camera  # Test sur flux caméra live
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
from pathlib import Path

from marker_detection import config
from marker_detection.runtime import create_aruco_detector, create_clahe
from marker_detection.detection import detect_aruco, validate_aruco, _preprocess_frame


def create_trackbars(window_name: str) -> dict:
    """Cree les trackbars pour interactivement changer parametres."""
    cv2.createTrackbar("CLAHE_CLIP", window_name, 40, 100,
                       lambda x: None)  # 0.4 to 10.0
    cv2.createTrackbar("CLAHE_TILE", window_name, 6, 16,
                       lambda x: None)  # 4 to 16
    cv2.createTrackbar("BLUR_SIZE", window_name, 5, 15,
                       lambda x: None)  # 3 to 15 (odd)
    cv2.createTrackbar("MIN_AREA", window_name, 80, 500,
                       lambda x: None)
    cv2.createTrackbar("MIN_STD (x10)", window_name, 20, 100,
                       lambda x: None)
    return {
        "clip": "CLAHE_CLIP",
        "tile": "CLAHE_TILE",
        "blur": "BLUR_SIZE",
        "area": "MIN_AREA",
        "std": "MIN_STD (x10)",
    }


def get_trackbar_values(window_name: str, trackbars: dict) -> dict:
    """Recupere les valeurs actuelles des trackbars."""
    return {
        "CLAHE_CLIP": cv2.getTrackbarPos(trackbars["clip"], window_name) / 10.0,
        "CLAHE_TILE": cv2.getTrackbarPos(trackbars["tile"], window_name),
        "BLUR_SIZE": cv2.getTrackbarPos(trackbars["blur"], window_name),
        "BLUR_SIZE": (
            cv2.getTrackbarPos(trackbars["blur"], window_name) * 2 + 1
        ),  # Force odd
        "MIN_AREA": cv2.getTrackbarPos(trackbars["area"], window_name),
        "MIN_STD": cv2.getTrackbarPos(trackbars["std"], window_name) / 10.0,
    }


def test_detection_on_image(image_path: str, interactive: bool = False) -> None:
    """Test detection sur une image statique."""
    print(f"Chargement image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Erreur: fichier non trouve ou non lisible")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = create_aruco_detector()

    if interactive:
        cv2.namedWindow("Detection Tuning")
        trackbars = create_trackbars("Detection Tuning")
        print("\n📝 Modifiez les trackbars pour tester. Appuyez sur Q pour quitter.\n")

    while True:
        # Valeurs par défaut ou trackbars
        if interactive:
            params = get_trackbar_values("Detection Tuning", trackbars)
            clahe_clip = params["CLAHE_CLIP"]
            clahe_tile = params["CLAHE_TILE"]
            blur_size = max(3, params["BLUR_SIZE"])
            if blur_size % 2 == 0:
                blur_size += 1
        else:
            clahe_clip = config.CLAHE_CLIP_LIMIT
            clahe_tile = config.CLAHE_TILE_GRID_SIZE[0]
            blur_size = 5

        # Recrée CLAHE avec params actuels
        clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                tileGridSize=(clahe_tile, clahe_tile))

        # Detecte
        corners, ids = detect_aruco(gray, detector, clahe)

        # Draw
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if ids is not None:
            for corner, mid in zip(corners, ids):
                pts = corner[0].astype(int)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                cv2.putText(display, f"ID:{mid}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                print(f"  → Détecté: ArUco {mid}")

        # Affiche params
        text_lines = [
            f"CLAHE: {clahe_clip:.1f} | Tile: {clahe_tile}x{clahe_tile}",
            f"Blur: {blur_size}x{blur_size}",
            f"Détectés: {len(ids or [])} ArUco",
        ]
        for i, line in enumerate(text_lines):
            cv2.putText(display, line, (20, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        if interactive:
            cv2.imshow("Detection Tuning", display)
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break
        else:
            cv2.imshow("Detection Result", display)
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()


def test_detection_on_camera(interactive: bool = True) -> None:
    """Test detection sur flux camera live."""
    from marker_detection.runtime import create_capture

    try:
        cap = create_capture()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    detector = create_aruco_detector()

    cv2.namedWindow("Camera - Detection Tuning")
    trackbars = create_trackbars("Camera - Detection Tuning")
    print("\n📝 Modifiez les trackbars. Appuyez sur Q pour quitter.\n")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erreur capture camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        # Trackbar values
        params = get_trackbar_values("Camera - Detection Tuning", trackbars)
        clahe_clip = params["CLAHE_CLIP"]
        clahe_tile = params["CLAHE_TILE"]
        blur_size = max(3, params["BLUR_SIZE"])
        if blur_size % 2 == 0:
            blur_size += 1

        clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                tileGridSize=(clahe_tile, clahe_tile))

        # Resize pour affichage
        display = cv2.resize(frame, (1280, 720))
        scale = 1280.0 / frame.shape[1]

        corners, ids = detect_aruco(gray, detector, clahe)

        if ids is not None:
            for corner, mid in zip(corners, ids):
                pts = (corner[0] * scale).astype(int)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                cv2.putText(display, f"ID:{mid}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        text_lines = [
            f"Frame: {frame_count} | FPS: {1000/max(1, cv2.getTickFrequency()):.1f}",
            f"CLAHE: {clahe_clip:.1f} | Tile: {clahe_tile}x{clahe_tile} | Blur: {blur_size}",
            f"Détectés: {len(ids or [])} ArUco",
        ]
        for i, line in enumerate(text_lines):
            cv2.putText(display, line, (20, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        cv2.imshow("Camera - Detection Tuning", display)

        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Outil de tuning interactif pour detection ArUco"
    )
    parser.add_argument("--image", type=str,
                        help="Chemin vers image statique a tester")
    parser.add_argument("--camera", action="store_true",
                        help="Tester sur camera live")
    parser.add_argument("--interactive", action="store_true", default=True,
                        help="Mode interactif avec trackbars (defaut)")
    parser.add_argument("--quiet", action="store_true",
                        help="Désactiver mode interactif")

    args = parser.parse_args()

    if args.image:
        test_detection_on_image(args.image, interactive=not args.quiet)
    elif args.camera:
        test_detection_on_camera(interactive=not args.quiet)
    else:
        parser.print_help()
