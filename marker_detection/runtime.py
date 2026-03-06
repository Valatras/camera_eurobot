"""Fonctions d'initialisation OpenCV (camera, fenetres, detecteurs)."""

from __future__ import annotations

import cv2

from marker_detection import config


def create_capture() -> cv2.VideoCapture:
    """Initialise la camera avec les parametres du projet."""
    candidates = [config.CAMERA_INDEX]
    candidates.extend(i for i in range(1,5) if i != config.CAMERA_INDEX)

    for camera_index in candidates:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        return cap

    raise RuntimeError(
        "Aucune camera accessible via V4L2 (index testes: "
        + ", ".join(str(i) for i in candidates)
        + ")."
    )


def create_windows() -> None:
    """Cree et redimensionne les fenetres d'affichage."""
    cv2.namedWindow(config.WINDOW_CAMERA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_CAMERA, *config.WINDOW_CAMERA_SIZE)

    cv2.namedWindow(config.WINDOW_AERIAL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_AERIAL, *config.WINDOW_AERIAL_SIZE)
    print("Fenetres creees: "
          f"{config.WINDOW_CAMERA} ({config.WINDOW_CAMERA_SIZE[0]}x{config.WINDOW_CAMERA_SIZE[1]}), "
          f"{config.WINDOW_AERIAL} ({config.WINDOW_AERIAL_SIZE[0]}x{config.WINDOW_AERIAL_SIZE[1]})")


def create_aruco_detector() -> cv2.aruco.ArucoDetector:
    """Construit un detecteur ArUco ajuste pour reduire les faux positifs."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.minMarkerPerimeterRate = 0.01
    aruco_params.maxMarkerPerimeterRate = 4.0
    aruco_params.polygonalApproxAccuracyRate = 0.04
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementWinSize = 5
    aruco_params.cornerRefinementMaxIterations = 30
    aruco_params.errorCorrectionRate = 0.5
    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def create_qr_detector() -> cv2.QRCodeDetector:
    """Construit le detecteur QR."""
    return cv2.QRCodeDetector()


def create_clahe() -> cv2.CLAHE:
    """Construit le pretraitement CLAHE utilise avant detection."""
    return cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
