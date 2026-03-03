"""Detection ArUco/QR avec validation et passe multi-resolution."""

from __future__ import annotations

import cv2
import numpy as np

from marker_detection import config


def validate_aruco(corner: np.ndarray, gray: np.ndarray) -> bool:
    """Valide un candidat ArUco pour eliminer les faux positifs grossiers."""
    pts = corner[0].astype(int)
    area = cv2.contourArea(pts)
    if area < 200 or area > 80000:
        return False

    x, y, w, h = cv2.boundingRect(pts)
    if h == 0:
        return False

    aspect = w / h
    if aspect < 0.3 or aspect > 3.0:
        return False

    bbox_area = w * h
    if bbox_area > 0:
        extent = area / bbox_area
        if extent < 0.4:
            return False

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    pixels = gray[mask > 0]
    if len(pixels) == 0:
        return False

    # Une variance trop faible indique souvent un faux marqueur uniforme.
    return float(np.std(pixels)) >= 35.0


def detect_all(
    frame_gray: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    qr_detector: cv2.QRCodeDetector,
    clahe: cv2.CLAHE,
) -> tuple[list[np.ndarray], list[int], list[str], list[np.ndarray]]:
    """Detecte ArUco et QR en combinant une passe rapide puis une passe de rattrapage."""
    small = cv2.resize(
        frame_gray,
        (
            int(frame_gray.shape[1] * config.DETECT_SCALE),
            int(frame_gray.shape[0] * config.DETECT_SCALE),
        ),
        interpolation=cv2.INTER_AREA,
    )
    enhanced_small = clahe.apply(small)

    aruco_corners: list[np.ndarray] = []
    aruco_ids: list[int] = []
    seen_aruco: set[int] = set()

    raw_corners, raw_ids, _ = detector.detectMarkers(enhanced_small)
    if raw_ids is not None:
        for corner, mid in zip(raw_corners, raw_ids.flatten()):
            marker_id = int(mid)
            corner_full = corner / config.DETECT_SCALE
            if validate_aruco(corner_full, frame_gray):
                aruco_corners.append(corner_full)
                aruco_ids.append(marker_id)
                seen_aruco.add(marker_id)

    found_corners = {mid for mid in aruco_ids if mid in config.CORNER_IDS}
    if found_corners != config.CORNER_IDS:
        enhanced_full = clahe.apply(frame_gray)
        raw_corners_full, raw_ids_full, _ = detector.detectMarkers(enhanced_full)
        if raw_ids_full is not None:
            for corner, mid in zip(raw_corners_full, raw_ids_full.flatten()):
                marker_id = int(mid)
                if marker_id not in seen_aruco and validate_aruco(corner, frame_gray):
                    aruco_corners.append(corner)
                    aruco_ids.append(marker_id)
                    seen_aruco.add(marker_id)

    qr_data: list[str] = []
    qr_corners: list[np.ndarray] = []
    retval, decoded, points, _ = qr_detector.detectAndDecodeMulti(enhanced_small)
    if retval and points is not None:
        for data, pts in zip(decoded, points):
            if data:
                qr_data.append(data)
                qr_corners.append(pts / config.DETECT_SCALE)

    return aruco_corners, aruco_ids, qr_data, qr_corners
