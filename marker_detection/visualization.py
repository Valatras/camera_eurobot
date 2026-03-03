"""Fonctions de rendu pour la vue camera et la vue aerienne."""

from __future__ import annotations

import cv2
import numpy as np

from marker_detection import config


def draw_grid(frame: np.ndarray, h_grid_to_img: np.ndarray | None) -> None:
    """Dessine la grille de jeu projetee dans l'image camera."""
    if h_grid_to_img is None:
        return

    for x in range(0, config.GRID_COLS + 1, 5):
        line_grid = np.float32([[[x, 0]], [[x, config.GRID_ROWS]]])
        line_img = cv2.perspectiveTransform(line_grid, h_grid_to_img).astype(np.int32)

        if config.ARUCO_OFFSET_COLS <= x <= (config.ARUCO_OFFSET_COLS + config.ARUCO_INNER_COLS):
            color = (0, 255, 0)
        else:
            color = (0, 150, 0)

        cv2.line(frame, tuple(line_img[0, 0]), tuple(line_img[1, 0]), color, 1)

    for y in range(0, config.GRID_ROWS + 1, 5):
        line_grid = np.float32([[[0, y]], [[config.GRID_COLS, y]]])
        line_img = cv2.perspectiveTransform(line_grid, h_grid_to_img).astype(np.int32)

        if config.ARUCO_OFFSET_ROWS <= y <= (config.ARUCO_OFFSET_ROWS + config.ARUCO_INNER_ROWS):
            color = (0, 255, 0)
        else:
            color = (0, 150, 0)

        cv2.line(frame, tuple(line_img[0, 0]), tuple(line_img[1, 0]), color, 1)


def draw_table_outline(frame: np.ndarray, table_pts: np.ndarray | None, aruco_pts: np.ndarray | None) -> None:
    """Dessine les contours de table et les points ArUco retenus."""
    if table_pts is not None:
        cv2.polylines(frame, [table_pts.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 2)

    if aruco_pts is None:
        return

    cv2.polylines(frame, [aruco_pts.astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 2)

    labels = ["TL(23)", "TR(22)", "BR(20)", "BL(21)"]
    colors = [(255, 0, 255), (255, 128, 0), (0, 255, 255), (128, 255, 0)]
    for pt, label, color in zip(aruco_pts, labels, colors):
        p = tuple(pt.astype(int))
        cv2.circle(frame, p, 8, color, -1)
        cv2.putText(frame, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_aerial_grid(aerial: np.ndarray) -> None:
    """Dessine la grille dans la vue aerienne."""
    cell_w = config.AERIAL_W / config.GRID_COLS
    cell_h = config.AERIAL_H / config.GRID_ROWS

    for col in range(0, config.GRID_COLS + 1, 5):
        x = int(col * cell_w)
        color = (0, 255, 0) if config.ARUCO_OFFSET_COLS <= col <= (config.ARUCO_OFFSET_COLS + config.ARUCO_INNER_COLS) else (0, 150, 0)
        cv2.line(aerial, (x, 0), (x, config.AERIAL_H), color, 1)
        if col % 10 == 0:
            cv2.putText(aerial, f"{col}", (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    for row in range(0, config.GRID_ROWS + 1, 5):
        y = int(row * cell_h)
        color = (0, 255, 0) if config.ARUCO_OFFSET_ROWS <= row <= (config.ARUCO_OFFSET_ROWS + config.ARUCO_INNER_ROWS) else (0, 150, 0)
        cv2.line(aerial, (0, y), (config.AERIAL_W, y), color, 1)
        if row % 5 == 0:
            cv2.putText(aerial, f"{row}", (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def draw_detection(frame: np.ndarray, pts_raw: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    """Dessine un polygone detecte et son libelle sur une image."""
    pts = pts_raw.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], True, color, 2)
    center = pts_raw.mean(axis=0).astype(int)
    cv2.putText(frame, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_aerial_detection(
    aerial: np.ndarray,
    pts_camera: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    h_img_to_aerial: np.ndarray | None,
) -> None:
    """Projette et dessine une detection camera sur la vue aerienne."""
    if h_img_to_aerial is None:
        return

    pts_aerial = cv2.perspectiveTransform(pts_camera.reshape(-1, 1, 2).astype(np.float32), h_img_to_aerial)
    pts_int = pts_aerial.astype(np.int32)

    cv2.polylines(aerial, [pts_int], True, color, 2)
    center = pts_aerial.mean(axis=0)[0].astype(int)
    cv2.putText(aerial, label, (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
