"""Fonctions d'initialisation OpenCV (camera, fenetres, detecteurs)."""

from __future__ import annotations

import cv2

from marker_detection import config


def create_capture() -> cv2.VideoCapture:
    """Initialise la camera (URL distante ou device V4L2 local)."""

    # Remote camera stream (MJPEG or RTSP URL)
    if config.CAMERA_URL:
        cap = cv2.VideoCapture(config.CAMERA_URL)
        if not cap.isOpened():
            raise RuntimeError(
                f"Impossible d'ouvrir le flux distant: {config.CAMERA_URL}"
            )
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # Local V4L2 device
    candidates = [config.CAMERA_INDEX]
    candidates.extend(i for i in range(0, 5) if i != config.CAMERA_INDEX)

    for camera_index in candidates:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    raise RuntimeError(
        "Aucune camera accessible via V4L2 (index testes: "
        + ", ".join(str(i) for i in candidates)
        + ")."
    )


def has_gui() -> bool:
    """Detecte si un backend GUI (GTK/Qt) est disponible."""
    try:
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        return True
    except cv2.error:
        return False


def create_windows(headless: bool = False) -> bool:
    """Cree les fenetres d'affichage. Retourne True si le GUI est actif."""
    if headless:
        return False

    if not has_gui():
        print("Pas de backend GUI disponible — mode headless automatique.")
        return False

    cv2.namedWindow(config.WINDOW_CAMERA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_CAMERA, *config.WINDOW_CAMERA_SIZE)

    cv2.namedWindow(config.WINDOW_AERIAL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_AERIAL, *config.WINDOW_AERIAL_SIZE)
    print("Fenetres creees: "
          f"{config.WINDOW_CAMERA} ({config.WINDOW_CAMERA_SIZE[0]}x{config.WINDOW_CAMERA_SIZE[1]}), "
          f"{config.WINDOW_AERIAL} ({config.WINDOW_AERIAL_SIZE[0]}x{config.WINDOW_AERIAL_SIZE[1]})")
    return True


def create_aruco_detector() -> cv2.aruco.ArucoDetector:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_params = cv2.aruco.DetectorParameters()

    # 🔥 CRUCIAL pour petits marqueurs
    aruco_params.minMarkerPerimeterRate = 0.01   # ↓↓↓
    aruco_params.maxMarkerPerimeterRate = 4.0

    # Adaptive threshold → plus fin
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23   # ↓↓↓
    aruco_params.adaptiveThreshWinSizeStep = 10

    # Moins strict géométriquement
    aruco_params.polygonalApproxAccuracyRate = 0.08

    # Corner refinement
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementWinSize = 7

    # Très important pour petits marqueurs
    aruco_params.minCornerDistanceRate = 0.005
    aruco_params.minDistanceToBorder = 1

    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def create_clahe() -> cv2.CLAHE:
    """Construit le pretraitement CLAHE utilise avant detection."""
    return cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
