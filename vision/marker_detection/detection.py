"""Detection ArUco avec validation et passe multi-resolution."""

from __future__ import annotations

import cv2
import numpy as np

from marker_detection import config


def validate_aruco(corner: np.ndarray, gray: np.ndarray, is_small: bool = False) -> bool:
    """Valide un candidat ArUco pour eliminer les faux positifs grossiers.

    Args:
        corner: Coins du marqueur
        gray: Image en niveaux de gris
        is_small: Si True, applique des critères plus lenients pour petits marqueurs
    """
    pts = corner[0].astype(int)
    area = cv2.contourArea(pts)

    # Critères d'aire + souples pour petits marqueurs
    min_area = 100 if is_small else 200
    max_area = 120000 if is_small else 80000
    if area < min_area or area > max_area:
        return False

    x, y, w, h = cv2.boundingRect(pts)
    if h == 0:
        return False

    # Aspect ratio + souple pour petits marqueurs bruitées
    aspect = w / h
    aspect_min = 0.25 if is_small else 0.3
    aspect_max = 5.0 if is_small else 3.0
    if aspect < aspect_min or aspect > aspect_max:
        return False

    bbox_area = w * h
    if bbox_area > 0:
        extent = area / bbox_area
        extent_min = 0.35 if is_small else 0.4  # Plus permissif pour le bruit
        if extent < extent_min:
            return False

    x, y = max(x, 0), max(y, 0)
    roi = gray[y:y+h, x:x+w]
    pts_local = pts - [x, y]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_local], 255)
    pixels = roi[mask > 0]
    if len(pixels) == 0:
        return False

    # Moins strict sur le contraste pour petits marqueurs lointains
    min_std = 30.0 if is_small else 35.0
    return float(np.std(pixels)) >= min_std


def detect_aruco(
    frame_gray: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    clahe: cv2.CLAHE,
    detect_scale: float = 1.0,
) -> tuple[list[np.ndarray], list[int]]:
    """Detecte ArUco avec strategie multi-echelle pour petits marqueurs.

    Passe 1 : pleine resolution (priorite sur petits marqueurs lointains).
    Passe 2 : detection a resolution reduite (detect_scale) pour performance.
    Passe 3 : si coins manquent, relance avec traitement d'amelioration du bruit.
    """
    aruco_corners: list[np.ndarray] = []
    aruco_ids: list[int] = []
    seen_aruco: set[int] = set()

    # ===== PASSE 0 : UPSCALE pour petits marqueurs =====
    upscale_factor = 3.0
    up = cv2.resize(frame_gray, None, fx=upscale_factor,
                    fy=upscale_factor, interpolation=cv2.INTER_LINEAR)

    up_enhanced = _preprocess_frame(up, clahe)

    raw_corners, raw_ids, _ = detector.detectMarkers(up_enhanced)

    if raw_ids is not None:
        inv_scale = 1.0 / upscale_factor
        for corner, mid in zip(raw_corners, raw_ids.flatten()):
            corner_full = corner * inv_scale
            if validate_aruco(corner_full, frame_gray, is_small=True):
                aruco_corners.append(corner_full)
                aruco_ids.append(int(mid))
                seen_aruco.add(int(mid))

    # ===== PASSE 1 : Pleine resolution (meilleur pour petits marqueurs lointains) =====
    full_enhanced = _preprocess_frame(frame_gray, clahe, blur_size=5)
    raw_corners, raw_ids, _ = detector.detectMarkers(full_enhanced)
    if raw_ids is not None:
        for corner, mid in zip(raw_corners, raw_ids.flatten()):
            marker_id = int(mid)
            if validate_aruco(corner, frame_gray, is_small=True):
                aruco_corners.append(corner)
                aruco_ids.append(marker_id)
                seen_aruco.add(marker_id)

    # ===== PASSE 2 : Resolution reduite (rapide, pour gros marqueurs) =====
    if detect_scale < 1.0:
        small = cv2.resize(frame_gray, None, fx=detect_scale, fy=detect_scale,
                           interpolation=cv2.INTER_AREA)
        small_enhanced = _preprocess_frame(small, clahe, blur_size=5)

        inv_scale = 1.0 / detect_scale
        raw_corners, raw_ids, _ = detector.detectMarkers(small_enhanced)
        if raw_ids is not None:
            for corner, mid in zip(raw_corners, raw_ids.flatten()):
                marker_id = int(mid)
                if marker_id not in seen_aruco:
                    corner_full = corner * inv_scale
                    if validate_aruco(corner_full, frame_gray, is_small=False):
                        aruco_corners.append(corner_full)
                        aruco_ids.append(marker_id)
                        seen_aruco.add(marker_id)

    # ===== PASSE 3 : Rattrapage avec debruitage pour coins manquants =====
    found_corners = {mid for mid in aruco_ids if mid in config.CORNER_IDS}
    if found_corners != config.CORNER_IDS:
        # Appliquer un debruitage bilateral pour ameliorer contraste sur le bruit
        denoised = cv2.bilateralFilter(frame_gray, 5, 75, 75)
        enhanced_denoised = _preprocess_frame(denoised, clahe, blur_size=3)

        raw_corners_full, raw_ids_full, _ = detector.detectMarkers(
            enhanced_denoised)
        if raw_ids_full is not None:
            for corner, mid in zip(raw_corners_full, raw_ids_full.flatten()):
                marker_id = int(mid)
                if marker_id not in seen_aruco and validate_aruco(corner, frame_gray, is_small=True):
                    aruco_corners.append(corner)
                    aruco_ids.append(marker_id)
                    seen_aruco.add(marker_id)

    return aruco_corners, aruco_ids


def _preprocess_frame(frame: np.ndarray, clahe: cv2.CLAHE, blur_size: int = 5) -> np.ndarray:
    return clahe.apply(frame)
