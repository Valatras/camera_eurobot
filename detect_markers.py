"""Point d'entree du pipeline de detection ArUco/QR."""

from __future__ import annotations

import os

# Sous Wayland (Hyprland), OpenCV/Qt cherche souvent le plugin "wayland"
# absent des wheels pip. Forcer XWayland (xcb) evite le crash GUI.
if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np

from marker_detection import config
from marker_detection.detection import detect_all
from marker_detection.geometry import build_transforms, to_cell
from marker_detection.runtime import (
    create_aruco_detector,
    create_capture,
    create_clahe,
    create_qr_detector,
    create_windows,
)
from marker_detection.tracking import Tracker
from marker_detection.visualization import (
    draw_aerial_detection,
    draw_aerial_grid,
    draw_detection,
    draw_grid,
    draw_table_outline,
)


def main() -> None:
    """Execute la boucle principale d'acquisition, detection et affichage."""
    try:
        cap = create_capture()
    except RuntimeError as exc:
        print(f"[ERREUR] {exc}")
        return
    create_windows()

    detector = create_aruco_detector()
    qr_detector = create_qr_detector()
    clahe = create_clahe()

    aruco_tracker = Tracker(buffer_size=3, min_hits=2)
    qr_tracker = Tracker(buffer_size=3, min_hits=2)

    last_h_aerial = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        a_corners, a_ids, q_data, q_corners = detect_all(gray, detector, qr_detector, clahe)
        a_ids, a_corners = aruco_tracker.update(a_ids, a_corners)
        q_data, q_corners = qr_tracker.update(q_data, q_corners)

        corners_by_id: dict[int, np.ndarray] = {}
        obj_aruco: list[tuple[int, np.ndarray]] = []
        for marker_id, corner in zip(a_ids, a_corners):
            if marker_id in config.CORNER_IDS:
                corners_by_id[marker_id] = corner
            else:
                obj_aruco.append((marker_id, corner))

        h_img_to_grid, h_grid_to_img, h_img_to_aerial, table_pts, aruco_pts = build_transforms(corners_by_id)

        if h_img_to_aerial is not None:
            last_h_aerial = h_img_to_aerial

        aerial = None
        if last_h_aerial is not None and frame_count % 2 == 0:
            aerial = cv2.warpPerspective(frame, last_h_aerial, (config.AERIAL_W, config.AERIAL_H))
            draw_aerial_grid(aerial)

        draw_grid(frame, h_grid_to_img)
        draw_table_outline(frame, table_pts, aruco_pts)

        for marker_id, corner in corners_by_id.items():
            draw_detection(frame, corner[0], f"C{marker_id}", (0, 255, 255))

        for marker_id, corner in obj_aruco:
            center = corner[0].mean(axis=0)
            pos = to_cell(center[0], center[1], h_img_to_grid)
            label = f"A{marker_id}[{pos[0]:.1f},{pos[1]:.1f}]" if pos else f"A{marker_id}"
            draw_detection(frame, corner[0], label, (0, 255, 0))

            if aerial is not None and frame_count % 2 == 0:
                draw_aerial_detection(aerial, corner[0], f"A{marker_id}", (0, 255, 0), last_h_aerial)

        for data, corners in zip(q_data, q_corners):
            center = corners.mean(axis=0)
            pos = to_cell(center[0], center[1], h_img_to_grid)
            short = data[:10] + "..." if len(data) > 10 else data
            label = f"QR:{short}[{pos[0]:.1f},{pos[1]:.1f}]" if pos else f"QR:{short}"
            draw_detection(frame, corners, label, (0, 165, 255))

            if aerial is not None and frame_count % 2 == 0:
                draw_aerial_detection(aerial, corners, f"QR:{short}", (0, 165, 255), last_h_aerial)

        n_corners = len(corners_by_id)
        missing = config.CORNER_IDS - set(corners_by_id)
        grid_ok = h_img_to_grid is not None

        cv2.putText(
            frame,
            f"C:{n_corners}/4 Obj:{len(obj_aruco) + len(q_data)} Grid:{'OK' if grid_ok else '...'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if grid_ok else (0, 165, 255),
            2,
        )
        if missing:
            cv2.putText(
                frame,
                f"Missing: {sorted(missing)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        cv2.imshow(config.WINDOW_CAMERA, frame)
        if aerial is not None:
            cv2.imshow(config.WINDOW_AERIAL, aerial)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
