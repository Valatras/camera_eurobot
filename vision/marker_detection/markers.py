"""Helpers pour classifier et filtrer les marqueurs detectes."""

from __future__ import annotations

import math

import cv2
import numpy as np

from marker_detection import config
from marker_detection.geometry import (
    compensate_height,
    project_marker_to_ground,
    to_cm,
)
from marker_detection.esp32_sender import ESP32Sender


def classify_marker_id(marker_id: int) -> str:
    """Mappe un id ArUco vers un type lisible."""
    if marker_id in config.CORNER_IDS:
        return f"TABLE{marker_id}"
    if marker_id == config.NUT_BLUE_ID:
        return "NUT_BLUE"
    if marker_id == config.NUT_YELLOW_ID:
        return "NUT_YELLOW"
    if marker_id == config.NUT_EMPTY_ID:
        return "NUT_EMPTY"
    if 1 <= marker_id <= 5:
        return f"BR{marker_id}"
    if 6 <= marker_id <= 10:
        return f"YR{marker_id - 5}"
    return f"ARUCO{marker_id}"


def is_opponent(marker_id: int) -> bool:
    """True si le marqueur appartient a un robot adverse."""
    if config.TEAM_COLOR == "blue":
        return 6 <= marker_id <= 10
    return 1 <= marker_id <= 5


def is_ally(marker_id: int) -> bool:
    """True si le marqueur appartient a un robot allie."""
    if config.TEAM_COLOR == "blue":
        return 1 <= marker_id <= 5
    return 6 <= marker_id <= 10


def separate_markers(
    a_ids: list[int],
    a_corners: list[np.ndarray],
) -> tuple[dict[int, np.ndarray], list[tuple[int, np.ndarray]]]:
    """Separe les coins de table des autres ArUco."""
    corners_by_id: dict[int, np.ndarray] = {}
    obj_aruco: list[tuple[int, np.ndarray]] = []

    for marker_id, corner in zip(a_ids, a_corners):
        if marker_id in config.CORNER_IDS:
            corners_by_id[marker_id] = corner
        else:
            obj_aruco.append((marker_id, corner))

    return corners_by_id, obj_aruco


def build_detected_list(
    corners_by_id: dict[int, np.ndarray],
    obj_aruco: list[tuple[int, np.ndarray]],
    h_img_to_cm: np.ndarray | None,
    *,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
    camera_rvec: np.ndarray | None = None,
    camera_tvec: np.ndarray | None = None,
    prefer_pnp: bool = True,
) -> list[tuple[str, float, float, float]]:
    """Construit la liste triee des marqueurs detectes en coords table (cm).

    Chemin sous-cm (prefere) : si K, dist, camera_rvec, camera_tvec sont
    fournis, on applique solvePnP par marqueur avec sa taille physique
    connue puis on projette le centre 3D sur le sol (z=0) par ray-casting
    depuis la camera. Biais de parallaxe = exact (pas de constante figee).

    Chemin legacy : si les intrinseques ne sont pas dispo, fallback sur
    l'homographie 2D + compensate_height (hauteurs et position camera
    codees en dur). Garde le pipeline fonctionnel meme sans calibration.

    Returns:
        Liste de tuples (label, x_cm, y_cm, angle_deg).
    """
    detected: list[tuple[str, float, float, float]] = []
    use_pnp = (
        K is not None
        and dist is not None
        and camera_rvec is not None
        and camera_tvec is not None
    )

    all_markers = list(corners_by_id.items()) + obj_aruco

    for marker_id, corner in all_markers:
        pos: tuple[float, float] | None = None
        angle: float | None = None

        if prefer_pnp and use_pnp:
            pos, angle = _solve_marker_pnp(
                marker_id, corner, K, dist, camera_rvec, camera_tvec,
            )

        if pos is None and not prefer_pnp and h_img_to_cm is not None:
            # Vue carte 2D: homographie seule, sans compensation de hauteur.
            # Cela évite d'introduire un effet de parallaxe dans le dashboard.
            center = corner[0].mean(axis=0)
            pos = to_cm(center[0], center[1], h_img_to_cm)
            if pos is None:
                continue
            angle = _compute_marker_angle_deg(corner, h_img_to_cm)

        if pos is None:
            # Fallback legacy : homographie + compensate_height.
            center = corner[0].mean(axis=0)
            pos = to_cm(center[0], center[1], h_img_to_cm)
            if pos is None:
                continue
            if marker_id in config.NUT_IDS:
                pos = compensate_height(
                    pos[0], pos[1],
                    config.CAMERA_TABLE_POSITION_CM,
                    config.NUT_HEIGHT_CM,
                    config.CAMERA_HEIGHT_CM,
                )
            elif 1 <= marker_id <= 10:
                pos = compensate_height(
                    pos[0], pos[1],
                    config.CAMERA_TABLE_POSITION_CM,
                    config.ROBOT_HEIGHT_CM,
                    config.CAMERA_HEIGHT_CM,
                )
            angle = _compute_marker_angle_deg(corner, h_img_to_cm)

        if angle is None:
            angle = _compute_marker_angle_deg(corner, h_img_to_cm)
        detected.append((classify_marker_id(marker_id), pos[0], pos[1], angle))

    detected.sort(key=lambda item: item[0])
    return detected


def _solve_marker_pnp(
    marker_id: int,
    corner: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    camera_rvec: np.ndarray,
    camera_tvec: np.ndarray,
) -> tuple[tuple[float, float] | None, float | None]:
    """Resout la pose d'un marqueur par solvePnP et projette au sol.

    Retourne ((x_cm, y_cm), angle_deg) ou (None, None) si echec.
    """
    size_mm = config.marker_size_mm(marker_id)
    height_cm = config.marker_height_cm(marker_id)
    # obj_pts dans le repere local du marqueur, a plat dans son propre plan
    # (z = 0). La hauteur physique sera encodee dans la pose table via le
    # ray-casting effectue par project_marker_to_ground.
    s = size_mm / 2.0
    obj_pts = np.array([
        [-s,  s, 0.0],
        [s,  s, 0.0],
        [s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float32)

    img_pts = np.asarray(corner, dtype=np.float32).reshape(-1, 2)
    if img_pts.shape[0] != 4:
        return None, None

    try:
        ok, rvec_m, tvec_m = cv2.solvePnP(
            obj_pts, img_pts, K, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return None, None
    if not ok:
        return None, None

    # tvec_m = position du centre du marqueur dans le repere camera (mm).
    # On ajoute la hauteur physique : le marqueur vu n'est pas au sol mais
    # a z=height_cm dans le repere table. On corrige la z-component du
    # centre dans le repere table en utilisant la pose camera.
    center_cam_mm = tvec_m.reshape(3).astype(np.float64)

    # Projection sur le plan table z=0 a partir de la camera, en tenant
    # compte de la hauteur du marqueur : on vise en realite le plan
    # z = height_cm*10 mm dans le repere table, mais on veut la position
    # du point au sol -> cela revient a avancer le centre dans le repere
    # table du vecteur (0,0,-height_mm) puis a ray-caster comme avant.
    x_cm, y_cm = project_marker_to_ground(
        center_cam_mm, camera_rvec, camera_tvec)

    # Correction hauteur : si marker est a z=h, le "point au sol" sous le
    # marqueur est deja son centre XY dans le repere table (projection
    # verticale), pas le point de visee ray-cast. Ici on veut la position
    # reelle de l'objet, pas le point vu. Le ray-casting precedent donne
    # la position projetee depuis la camera ; pour obtenir le point au
    # sol directement sous le marqueur a z=h il faut avancer de h/h_cam *
    # distance supplementaire vers le nadir. On ajuste en re-calculant
    # avec un plan z=h_mm au lieu de z=0.
    if height_cm > 0.0:
        x_cm, y_cm = _project_marker_at_height(
            center_cam_mm, camera_rvec, camera_tvec, height_cm,
        )

    # Angle : extrait du rvec marqueur compose avec la pose camera.
    angle_deg = _marker_yaw_in_table(rvec_m, camera_rvec)
    return (x_cm, y_cm), angle_deg


def _project_marker_at_height(
    marker_center_cam_mm: np.ndarray,
    camera_rvec: np.ndarray,
    camera_tvec: np.ndarray,
    height_cm: float,
) -> tuple[float, float]:
    """Variante de project_marker_to_ground pour un marqueur a z = height_cm.

    Le marqueur observe est a z = height_mm dans le repere table : sa
    position XY au sol est simplement sa position XY dans ce plan (pas
    besoin de ray-casting depuis la camera). On transforme tvec_marqueur
    vers le repere table et on ignore la z-component.
    """
    R_cam, _ = cv2.Rodrigues(camera_rvec)
    t = np.asarray(camera_tvec, dtype=np.float64).reshape(3, 1)
    p_cam = np.asarray(marker_center_cam_mm, dtype=np.float64).reshape(3, 1)
    p_table = (R_cam.T @ (p_cam - t)).flatten()
    x_cm = p_table[0] / 10.0
    y_cm = p_table[1] / 10.0
    return x_cm, y_cm


def _marker_yaw_in_table(
    marker_rvec: np.ndarray,
    camera_rvec: np.ndarray,
) -> float:
    """Extrait l'angle de yaw (degres) d'un marqueur dans le repere table.

    L'orientation finale du marqueur dans le repere table est la
    composition R_table_marker = R_table_cam @ R_cam_marker. On extrait
    le yaw = atan2(R[1,0], R[0,0]) projete dans le plan xy.
    """
    R_cam_marker, _ = cv2.Rodrigues(marker_rvec)
    R_cam_table, _ = cv2.Rodrigues(camera_rvec)
    R_table_marker = R_cam_table.T @ R_cam_marker
    yaw = math.atan2(R_table_marker[1, 0], R_table_marker[0, 0])
    return float(math.degrees(yaw))


def print_detected_objects(
    corners_by_id: dict[int, np.ndarray],
    obj_aruco: list[tuple[int, np.ndarray]],
    h_img_to_cm: np.ndarray | None,
) -> None:
    """Imprime les objets detectes en coordonnees table (cm)."""
    detected = build_detected_list(corners_by_id, obj_aruco, h_img_to_cm)

    if detected:
        print(detected)


def send_detected_objects(
    corners_by_id: dict[int, np.ndarray],
    obj_aruco: list[tuple[int, np.ndarray]],
    h_img_to_cm: np.ndarray | None,
    sender: ESP32Sender,
) -> bool:
    """Envoie les objets detectes a un ESP32 via USB serie.

    Returns:
        True si l'envoi a reussi, False sinon.
    """
    detected = build_detected_list(corners_by_id, obj_aruco, h_img_to_cm)
    if not detected:
        return True
    return sender.send_markers(detected)


def _compute_marker_angle_deg(
    corner: np.ndarray,
    h_img_to_cm: np.ndarray | None,
) -> float:
    """Calcule l'inclinaison du marqueur en degres dans le repere table.

    L'angle est calcule entre le bord haut (coin 0 -> coin 1) et l'axe X.
    Valeur positive = rotation anti-horaire.
    """
    pts = corner[0].astype(np.float32)

    if h_img_to_cm is not None:
        pts = cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2), h_img_to_cm).reshape(-1, 2)
        pts[:, 0] = config.TABLE_W_CM - pts[:, 0]

    vec = pts[1] - pts[0]
    return float(math.degrees(math.atan2(vec[1], vec[0])))
