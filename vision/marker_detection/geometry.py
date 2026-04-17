"""Geometrie de la table: selection de coins et matrices de perspective.

Supporte la calibration avec 3 ou 4 marqueurs de coins.
Avec 3 coins : le 4e est extrapole par parallelogramme.
"""

from __future__ import annotations

import cv2
import numpy as np

from marker_detection import config

# Mapping marker_id -> position label
_CORNER_POSITIONS: dict[int, str] = dict(
    zip(config.CORNER_ORDER, ["TL", "TR", "BR", "BL"]))


def select_table_corner_point(marker_pts: np.ndarray, position: str) -> np.ndarray:
    """Choisit le coin du marqueur qui correspond au coin reel de table."""
    if position == "TL":
        idx = int(np.argmin(marker_pts[:, 0] + marker_pts[:, 1]))
    elif position == "TR":
        idx = int(np.argmin(marker_pts[:, 1] - marker_pts[:, 0]))
    elif position == "BR":
        idx = int(np.argmax(marker_pts[:, 0] + marker_pts[:, 1]))
    else:  # BL
        idx = int(np.argmax(marker_pts[:, 1] - marker_pts[:, 0]))

    return marker_pts[idx]


def _marker_center(corners: np.ndarray) -> np.ndarray:
    """Retourne le centre du marqueur ArUco dans l'image."""
    return corners.mean(axis=0)


def _flip_table_x(coords: tuple[float, float]) -> np.ndarray:
    """Retourne les coordonnees utilisees par le repere interne du pipeline."""
    x_cm, y_cm = coords
    return np.array([config.TABLE_W_CM - x_cm, y_cm], dtype=np.float32)


def _marker_grid_point(marker_id: int) -> np.ndarray:
    x_cm, y_cm = config.CORNER_REAL_POSITIONS_CM[marker_id]
    return np.array([config.GRID_COLS - x_cm / 10.0, y_cm / 10.0], dtype=np.float32)


def _build_projection(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray | None:
    if src_pts.shape[0] == 4:
        return cv2.getPerspectiveTransform(src_pts, dst_pts)

    affine, _ = cv2.estimateAffine2D(src_pts, dst_pts)
    if affine is None:
        return None
    return np.vstack([affine, [0.0, 0.0, 1.0]])


def compensate_height(
    x_cm: float,
    y_cm: float,
    camera_pos_cm: tuple[float, float],
    object_height_cm: float,
    camera_height_cm: float,
) -> tuple[float, float]:
    """Compense la projection d'un objet eleve au-dessus de la table.

    La detection ArUco se fait sur le haut de la noisette ;
    le repere table correspond au sol. Cette correction ramene la
    position vers le point au sol en fonction de la hauteur de la camera.
    """
    if camera_height_cm <= object_height_cm or object_height_cm <= 0.0:
        return x_cm, y_cm

    scale = 1.0 - object_height_cm / camera_height_cm
    return (
        camera_pos_cm[0] + (x_cm - camera_pos_cm[0]) * scale,
        camera_pos_cm[1] + (y_cm - camera_pos_cm[1]) * scale,
    )


def select_table_points(
    corners_by_id: dict[int, np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Selectionne les points ArUco detectes et leurs positions reels connues."""
    detected = [mid for mid in config.CORNER_ORDER if mid in corners_by_id]
    n_detected = len(detected)

    if n_detected < 3:
        return None, None, None, n_detected

    img_pts = np.array(
        [_marker_center(corners_by_id[mid][0]) for mid in detected],
        dtype=np.float32,
    )
    table_cm_pts = np.array(
        [_flip_table_x(config.CORNER_REAL_POSITIONS_CM[mid])
         for mid in detected],
        dtype=np.float32,
    )
    grid_pts = np.array(
        [_marker_grid_point(mid) for mid in detected],
        dtype=np.float32,
    )

    return img_pts, table_cm_pts, grid_pts, n_detected


def extrapolate_table_corners(aruco_pts: np.ndarray) -> np.ndarray:
    """Extrapole les 4 coins reels de table depuis la zone ArUco interieure."""
    top_vec = aruco_pts[1] - aruco_pts[0]
    right_vec = aruco_pts[2] - aruco_pts[1]
    bottom_vec = aruco_pts[2] - aruco_pts[3]
    left_vec = aruco_pts[3] - aruco_pts[0]

    width_ratio = config.TABLE_W_MM / \
        (config.TABLE_W_MM - 2 * config.ARUCO_INSET_MM)
    height_ratio = config.TABLE_H_MM / \
        (config.TABLE_H_MM - 2 * config.ARUCO_INSET_MM)

    width_ext = (width_ratio - 1) / 2
    height_ext = (height_ratio - 1) / 2

    table_tl = aruco_pts[0] - width_ext * top_vec - height_ext * left_vec
    table_tr = aruco_pts[1] + width_ext * top_vec - height_ext * right_vec
    table_br = aruco_pts[2] + width_ext * bottom_vec + height_ext * right_vec
    table_bl = aruco_pts[3] - width_ext * bottom_vec + height_ext * left_vec

    return np.array([table_tl, table_tr, table_br, table_bl], dtype=np.float32)


def _validate_homography_rms(
    h_img_to_cm: np.ndarray,
    img_pts: np.ndarray,
    table_cm_pts: np.ndarray,
) -> float:
    """Reprojette les coins detectes et calcule l'erreur RMS en cm.

    Sert a rejeter les homographies grossierement fausses (fausse detection
    d'un coin). Les coordonnees table etant deja "flippees" via _flip_table_x
    pour table_cm_pts, on compare dans ce meme repere sans ajouter d'inversion.
    """
    projected = cv2.perspectiveTransform(
        img_pts.reshape(-1, 1, 2).astype(np.float32), h_img_to_cm,
    ).reshape(-1, 2)
    diffs = projected - table_cm_pts
    return float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))


def build_transforms(
    corners_by_id: dict[int, np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Construit les matrices de perspective.

    La calibration utilise les positions reelles des quatre ArUco de table.
    h_img_to_cm donne des coordonnees table en cm pour les objets detectes.
    h_img_to_aerial conserve l'orientation naturelle de la camera.

    Rejette la calibration si la reprojection des coins detectes s'ecarte de
    plus de config.HOMOGRAPHY_MAX_RMS_CM (protection contre une fausse
    detection de coin qui corromprait toute la frame).
    """
    img_pts, table_cm_pts, grid_pts, n_corners = select_table_points(
        corners_by_id)
    if img_pts is None:
        return None, None, None, None, None, None, n_corners

    h_img_to_cm = _build_projection(img_pts, table_cm_pts)
    if h_img_to_cm is None:
        return None, None, None, None, None, None, n_corners

    # Validation par reprojection : si un coin a ete mal detecte, l'erreur
    # explose et on prefere rejeter plutot que publier des coords fausses.
    rms_cm = _validate_homography_rms(h_img_to_cm, img_pts, table_cm_pts)
    if rms_cm > config.HOMOGRAPHY_MAX_RMS_CM:
        print(
            f"[WARN] Homographie rejetee : RMS reprojection = {rms_cm:.2f} cm "
            f"(> {config.HOMOGRAPHY_MAX_RMS_CM:.1f} cm, n_corners={n_corners})"
        )
        return None, None, None, None, None, None, n_corners

    h_img_to_grid = _build_projection(img_pts, grid_pts)
    if h_img_to_grid is None:
        return None, None, None, None, None, None, n_corners

    h_grid_to_img = np.linalg.inv(h_img_to_grid)
    h_cm_to_img = np.linalg.inv(h_img_to_cm)

    play_cm_pts = np.float32(
        [[0.0, 0.0], [config.TABLE_W_CM, 0.0],
         [config.TABLE_W_CM, config.PLAY_H_CM], [0.0, config.PLAY_H_CM]]
    )
    aerial_px_pts = np.float32(
        [[0.0, 0.0], [config.AERIAL_W, 0.0],
         [config.AERIAL_W, config.AERIAL_H], [0.0, config.AERIAL_H]]
    )

    play_img_pts = cv2.perspectiveTransform(
        play_cm_pts.reshape(-1, 1, 2), h_cm_to_img,
    ).reshape(-1, 2).astype(np.float32)
    h_img_to_aerial = cv2.getPerspectiveTransform(play_img_pts, aerial_px_pts)

    return h_img_to_grid, h_grid_to_img, h_img_to_aerial, h_img_to_cm, img_pts, img_pts, n_corners


def to_cell(px: float, py: float, h_img_to_grid: np.ndarray | None) -> tuple[float, float] | None:
    """Projette un point image dans le repere grille (colonne, ligne)."""
    if h_img_to_grid is None:
        return None

    out = cv2.perspectiveTransform(np.float32([[[px, py]]]), h_img_to_grid)
    gx, gy = float(out[0, 0, 0]), float(out[0, 0, 1])
    # Inverser l'axe horizontal pour top-right = (0,0)
    gx = config.GRID_COLS - gx

    if gx < -0.5 or gx > config.GRID_COLS + 0.5 or gy < -0.5 or gy > config.GRID_ROWS + 0.5:
        return None

    return gx, gy


def to_cm(px: float, py: float, h_img_to_cm: np.ndarray | None) -> tuple[float, float] | None:
    """Projette un point image en coordonnees table (cm)."""
    if h_img_to_cm is None:
        return None

    out = cv2.perspectiveTransform(np.float32([[[px, py]]]), h_img_to_cm)
    x_cm, y_cm = float(out[0, 0, 0]), float(out[0, 0, 1])
    x_cm = config.TABLE_W_CM - x_cm  # Inverser l'axe horizontal

    if x_cm < -5 or x_cm > config.TABLE_W_CM + 5 or y_cm < -5 or y_cm > config.TABLE_H_CM + 5:
        return None

    return x_cm, y_cm


def compute_angle(corners: np.ndarray) -> float:
    """Calcule l'orientation d'un marqueur en degres depuis ses 4 coins."""
    pts = corners[0]
    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    return float(np.degrees(np.arctan2(dy, dx)))


# ---------------------------------------------------------------------------
# Auto-calibration extrinseque (pose camera dans le repere table)
# ---------------------------------------------------------------------------


def estimate_camera_pose(
    corners_by_id: dict[int, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estime rvec, tvec de la camera dans le repere table (en mm).

    Utilise les 4x4 = 16 coins des ArUco de table avec leurs positions 3D
    connues. Plus robuste que solvePnP sur un seul marqueur (meilleure
    geometrie pour l'estimation de pose).

    Retourne (rvec, tvec) ou None si l'estimation echoue.
    """
    from marker_detection.auto_calibration import corner_object_points_table

    obj_pts = []
    img_pts = []
    for cid in sorted(config.CORNER_IDS):
        if cid not in corners_by_id:
            continue
        pts = np.asarray(corners_by_id[cid], dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] != 4:
            continue
        obj_pts.append(corner_object_points_table(cid))
        img_pts.append(pts)

    if len(obj_pts) < 2:  # au moins 2 marqueurs = 8 points
        return None

    obj = np.concatenate(obj_pts, axis=0)
    img = np.concatenate(img_pts, axis=0)
    ok, rvec, tvec = cv2.solvePnP(
        obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rvec, tvec


def project_marker_to_ground(
    marker_center_cam_mm: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[float, float]:
    """Projette le centre 3D d'un marqueur sur le plan table (z=0).

    `marker_center_cam_mm` est le centre du marqueur dans le repere camera
    (tvec de solvePnP du marqueur objet). La pose (rvec, tvec) est celle
    de la camera dans le repere table. On transforme le centre marqueur
    vers le repere table puis on projette par ray-casting sur z=0 depuis
    le centre optique de la camera.

    Retourne (x_cm, y_cm) au sol dans le repere table reel (0 = bord
    gauche, TABLE_W_CM = bord droit), meme convention que le dashboard.
    """
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    p_cam = np.asarray(marker_center_cam_mm, dtype=np.float64).reshape(3, 1)
    # Position marqueur dans le repere table (mm) :
    p_table = (R.T @ (p_cam - t)).flatten()
    # Position camera dans le repere table (mm) :
    cam_table = (-R.T @ t).flatten()
    # Ray-cast : cam + t * (p - cam) = sol (z=0).
    dz = p_table[2] - cam_table[2]
    if abs(dz) < 1e-6:
        x_mm, y_mm = p_table[0], p_table[1]
    else:
        t_ray = -cam_table[2] / dz
        x_mm = cam_table[0] + t_ray * (p_table[0] - cam_table[0])
        y_mm = cam_table[1] + t_ray * (p_table[1] - cam_table[1])

    x_cm = x_mm / 10.0
    y_cm = y_mm / 10.0
    return x_cm, y_cm


def camera_extrinsics_summary(rvec: np.ndarray, tvec: np.ndarray) -> dict:
    """Resume la pose camera en valeurs lisibles pour le log / debug."""
    R, _ = cv2.Rodrigues(rvec)
    cam_table_mm = (-R.T @ tvec).flatten()
    return {
        "cam_x_cm": float(cam_table_mm[0] / 10.0),
        "cam_y_cm": float(cam_table_mm[1] / 10.0),
        "cam_height_cm": float(cam_table_mm[2] / 10.0),
    }
