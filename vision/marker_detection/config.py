"""Configuration statique du pipeline de detection."""

from __future__ import annotations

import os

FRAME_W = 3840
FRAME_H = 2160
CAMERA_INDEX = 2
CAMERA_FPS = 30
# URL of a remote MJPEG/RTSP stream (overrides CAMERA_INDEX when set).
CAMERA_URL = os.environ.get("CAMERA_URL", "")

WINDOW_CAMERA = "Camera"
WINDOW_AERIAL = "Aerial View"
WINDOW_CAMERA_SIZE = (3840, 2160)
WINDOW_AERIAL_SIZE = (800, 600)

CORNER_IDS = {20, 21, 22, 23}
# Ordre de reference pour la table: TL, TR, BR, BL
CORNER_ORDER = [23, 22, 20, 21]
# Positions reelles des centres des 4 codes ArUco de calibrage sur la table, en cm.
# Ces points sont utilises pour calculer l'homographie et creer les coords des autres objets.
CORNER_REAL_POSITIONS_CM = {
    20: (60.0, 140.0),
    21: (240.0, 140.0),
    22: (60.0, 60.0),
    23: (240.0, 60.0),
}

# Correction de hauteur pour la localisation des objets physiques.
# Ces constantes restent comme valeurs par defaut mais sont normalement
# ecrasees a chaque session par l'auto-calibration extrinseque (solvePnP
# sur les 4 coins de table), voir `auto_calibration.py`.
CAMERA_HEIGHT_CM = 133.0
NUT_HEIGHT_CM = 3.0
ROBOT_HEIGHT_CM = 45.0
CAMERA_TABLE_POSITION_CM = (160.0, 200.0)

# Tailles physiques des ArUco (mm), confirmees par reglement + mesure robot.
# Utilisees pour solvePnP: 1 mm d'erreur = ~3 mm de biais position.
MARKER_SIZE_NUT_MM = 40.0
MARKER_SIZE_ROBOT_MM = 70.0
MARKER_SIZE_CORNER_MM = 100.0

# Hauteurs physiques des marqueurs (cm) : distance du plan du marqueur
# au-dessus du sol de la table.
MARKER_HEIGHT_NUT_CM = 3.0
MARKER_HEIGHT_ROBOT_CM = 45.0
MARKER_HEIGHT_CORNER_CM = 0.0


def marker_size_mm(marker_id: int) -> float:
    """Taille physique (mm) du marqueur ArUco pour solvePnP."""
    if marker_id in CORNER_IDS:
        return MARKER_SIZE_CORNER_MM
    if marker_id in NUT_IDS:
        return MARKER_SIZE_NUT_MM
    if 1 <= marker_id <= 10:
        return MARKER_SIZE_ROBOT_MM
    # Fallback : taille noisette (le plus courant hors robot/coin).
    return MARKER_SIZE_NUT_MM


def marker_height_cm(marker_id: int) -> float:
    """Hauteur (cm) du plan du marqueur au-dessus de la table."""
    if marker_id in CORNER_IDS:
        return MARKER_HEIGHT_CORNER_CM
    if marker_id in NUT_IDS:
        return MARKER_HEIGHT_NUT_CM
    if 1 <= marker_id <= 10:
        return MARKER_HEIGHT_ROBOT_CM
    return MARKER_HEIGHT_NUT_CM


# IDs ArUco des caisses de noisettes (reglement §D.4)
NUT_BLUE_ID = 36
NUT_YELLOW_ID = 47
NUT_EMPTY_ID = 41
NUT_IDS = {NUT_BLUE_ID, NUT_YELLOW_ID, NUT_EMPTY_ID}

# Couleur d'equipe pour determiner adversaires ("blue" ou "yellow")
TEAM_COLOR = os.environ.get("TEAM_COLOR", "yellow")

TABLE_W_MM = 3000
TABLE_H_MM = 2000
TABLE_W_CM = 300
TABLE_H_CM = 200
ARUCO_INSET_MM = 600

# Zone de jeu visible (excluant grenier et nids, y=155..200 cm).
PLAY_H_CM = 155

GRID_COLS = 30
GRID_ROWS = 20

ARUCO_OFFSET_COLS = 5
ARUCO_OFFSET_ROWS = 5
ARUCO_INNER_COLS = 20
ARUCO_INNER_ROWS = 10

# Vue aerienne de la zone de jeu (300x155 cm, ratio ~1.94, ~3 px/cm).
AERIAL_W = 900
AERIAL_H = 465

# Validation d'homographie : RMS max de reprojection des coins connus (cm).
# Au-dessus, la calibration est rejetee (faux positif sur un coin probable).
HOMOGRAPHY_MAX_RMS_CM = 1.0

# Validation de l'auto-calibration intrinseque : RMS max (pixels) de
# reprojection des 16 points de coins ArUco avec la matrice K par defaut.
# Au-dela, l'auto-calibration echoue et demande un damier.
INTRINSIC_MAX_RMS_PX = 1.0

# Ratio focal par defaut (fx / image_width) pour une camera smartphone
# grand-angle typique en 4K. Utilise en premiere approximation si aucun
# fichier intrinsics.npz n'est present.
INTRINSIC_DEFAULT_FOCAL_RATIO = 0.85

# Warm-up calibration : nombre de frames consecutives avec 4 coins visibles
# requises avant de demarrer le pipeline principal. Echec si non atteint
# dans CALIBRATION_TIMEOUT_S.
CALIBRATION_MIN_FRAMES = 10
CALIBRATION_TIMEOUT_S = 15.0

# Duree de vie maximale du cache d'homographie (secondes). Au-dela, la calib
# cachee est consideree comme perimee : les donnees restent emises mais
# avec le flag "stale" afin que Dave sache qu'elles datent d'avant un
# possible deplacement de la camera.
HOMOGRAPHY_CACHE_TTL_S = 5.0

# Lissage temporel EMA applique aux positions (x, y, angle) projetees en cm.
# alpha = poids de la nouvelle mesure (1.0 = pas de lissage, 0.0 = gele).
EMA_ALPHA_POSITION = 0.4
EMA_ALPHA_ANGLE = 0.4
# Au-dela de ce timeout, le lissage est reinitialise (le marqueur a disparu).
EMA_RESET_TIMEOUT_S = 0.3

# ---------------------------------------------------------------------------
# Zones de ramassage (ZR) et garde-mangers (GM) - source: rules/dimensions.md
# ---------------------------------------------------------------------------
# Pour chaque ZR: centre (cm), axe long ("x" ou "y"), dimensions (cm).
# L'axe long definit dans quelle direction les 5 noisettes sont alignees.
# Toutes les ZR font 20x15 cm, mais l'axe long varie.
#
# Numerotation (rules/dimensions.md) :
#   1: (17.5, 120)  axe y   | 2: (282.5, 120) axe y
#   3: (115, 80)    axe x   | 4: (185, 80)    axe x
#   5: (17.5, 40)   axe y   | 6: (282.5, 40)  axe y
#   7: (110, 17.5)  axe x   | 8: (190, 17.5)  axe x
PICKUP_ZONES: list[dict] = [
    {"id": 1, "cx_cm": 17.5,  "cy_cm": 120.0,
        "axis": "y", "len_cm": 20.0, "wid_cm": 15.0},
    {"id": 2, "cx_cm": 282.5, "cy_cm": 120.0,
        "axis": "y", "len_cm": 20.0, "wid_cm": 15.0},
    {"id": 3, "cx_cm": 115.0, "cy_cm": 80.0,
        "axis": "x", "len_cm": 20.0, "wid_cm": 15.0},
    {"id": 4, "cx_cm": 185.0, "cy_cm": 80.0,
        "axis": "x", "len_cm": 20.0, "wid_cm": 15.0},
    {"id": 5, "cx_cm": 17.5,  "cy_cm": 40.0,
        "axis": "y", "len_cm": 20.0, "wid_cm": 15.0},
    {"id": 6, "cx_cm": 282.5, "cy_cm": 40.0,
        "axis": "y", "len_cm": 20.0, "wid_cm": 15.0},
    {"id": 7, "cx_cm": 110.0, "cy_cm": 17.5,
        "axis": "x", "len_cm": 20.0, "wid_cm": 15.0},
    {"id": 8, "cx_cm": 190.0, "cy_cm": 17.5,
        "axis": "x", "len_cm": 20.0, "wid_cm": 15.0},
]

# Tolerance d'appartenance a une zone (cm): un ArUco legerement deborde
# reste attribue a la zone.
ZONE_MEMBERSHIP_MARGIN_CM = 2.0

# Garde-mangers : 10 zones 20x20 cm (rules/dimensions.md).
GARDE_MANGERS: list[dict] = [
    {"id": 1,  "cx_cm": 125.0, "cy_cm": 145.0, "size_cm": 20.0},
    {"id": 2,  "cx_cm": 175.0, "cy_cm": 145.0, "size_cm": 20.0},
    {"id": 3,  "cx_cm": 10.0,  "cy_cm": 80.0,  "size_cm": 20.0},
    {"id": 4,  "cx_cm": 80.0,  "cy_cm": 80.0,  "size_cm": 20.0},
    {"id": 5,  "cx_cm": 150.0, "cy_cm": 80.0,  "size_cm": 20.0},
    {"id": 6,  "cx_cm": 220.0, "cy_cm": 80.0,  "size_cm": 20.0},
    {"id": 7,  "cx_cm": 290.0, "cy_cm": 80.0,  "size_cm": 20.0},
    {"id": 8,  "cx_cm": 70.0,  "cy_cm": 10.0,  "size_cm": 20.0},
    {"id": 9,  "cx_cm": 150.0, "cy_cm": 10.0,  "size_cm": 20.0},
    {"id": 10, "cx_cm": 230.0, "cy_cm": 10.0,  "size_cm": 20.0},
]

# Nombre max de noisettes empilees par ZR (5 emplacements theoriques).
ZONE_MAX_SLOTS = 5

# Duree au-dela de laquelle une observation stockee dans l'historique ZR
# est consideree obsolete (le robot a pu passer et tout bouger).
ZONE_HISTORY_TTL_S = 30.0

CLAHE_CLIP_LIMIT = 4.0  # Augmente : meilleur contraste pour petits marqueurs bruites
CLAHE_TILE_GRID_SIZE = (6, 6)  # Reduit : localise mieux le contraste

# Facteur de reduction pour la detection ArUco.
# 0.5 = detecte a 960x540 pour du 1080p (ou 1920x1080 pour du 4K).
DETECT_SCALE = float(os.environ.get("DETECT_SCALE", "0.5"))

# Taille d'affichage (resize avant imshow pour performance).
DISPLAY_W = 960
DISPLAY_H = 540

# Dashboard (web UI) Socket.IO server URL.
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:3001")

# MJPEG stream server port (for dashboard video feed).
MJPEG_PORT = int(os.environ.get("MJPEG_PORT", "8081"))
MJPEG_QUALITY = int(os.environ.get("MJPEG_QUALITY", "70"))
