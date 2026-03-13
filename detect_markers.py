"""Point d'entree du pipeline de detection ArUco/QR."""

from __future__ import annotations

import cv2

from marker_detection import config
from marker_detection.detection import detect_all
from marker_detection.geometry import build_transforms
from marker_detection.markers import print_detected_objects, separate_markers
from marker_detection.runtime import (
    create_aruco_detector,
    create_capture,
    create_clahe,
    create_qr_detector,
    create_windows,
)
from marker_detection.tracking import Tracker
from marker_detection.visualization import (
    compute_aerial,
    draw_corner_markers,
    draw_grid,
    draw_object_markers,
    draw_qr_codes,
    draw_status,
    draw_table_outline,
)
from marker_detection.esp32_sender import ESP32Sender
from marker_detection.markers import send_detected_objects


def main() -> None:
    """Boucle principale du pipeline."""

    # Ouvre une connexion camera et esp32.
    try:
        cap = create_capture()
        sender = ESP32Sender()
        sender.connect()
    except RuntimeError as exc:
        print(f"[ERREUR] {exc}")
        return

    create_windows()  # Ouverture des fenetres camera / aerienne.

    detector = create_aruco_detector()
    qr_detector = create_qr_detector()
    clahe = create_clahe()  # Pre-traitement pour stabiliser la detection.

    # Lissage temporel pour eviter les detections intermittentes.
    # Par le parametre min_hits, on peut forcer a attendre plusieurs detections coherentes
    # avant de valider un marqueur.
    aruco_tracker = Tracker(buffer_size=3, min_hits=2)
    qr_tracker = Tracker(buffer_size=3, min_hits=2)

    last_h_aerial = None
    frame_count = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # Detecteurs en niveaux de gris.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detection ArUco + QR (multi-resolution).
        a_corners, a_ids, q_data, q_corners = detect_all(
            gray, detector, qr_detector, clahe)

        # Filtrage temporel des detections.
        a_ids, a_corners = aruco_tracker.update(a_ids, a_corners)
        q_data, q_corners = qr_tracker.update(q_data, q_corners)

        # Separation des coins de table vs marqueurs objets.
        corners_by_id, obj_aruco = separate_markers(a_ids, a_corners)

        # Calcul des homographies et des points de table.
        h_img_to_grid, h_grid_to_img, h_img_to_aerial, table_pts, aruco_pts = build_transforms(
            corners_by_id)

        if h_img_to_aerial is not None:
            # Garder la derniere vue aerienne valide.
            last_h_aerial = h_img_to_aerial

        # Vue aerienne (une frame sur deux).
        aerial = compute_aerial(frame, last_h_aerial, frame_count)

        # Rendu overlays.
        draw_grid(frame, h_grid_to_img)
        draw_table_outline(frame, table_pts, aruco_pts)

        draw_corner_markers(frame, corners_by_id)
        draw_object_markers(
            frame, aerial, obj_aruco, h_img_to_grid, last_h_aerial, frame_count
        )

        draw_qr_codes(
            frame, aerial, q_data, q_corners, h_img_to_grid, last_h_aerial, frame_count
        )

        # envoies des donnees detectees dans l'ESP32 (ou la console si pas de connexion).
        if h_img_to_grid is not None and frame_count % 2 == 0:
            # print_detected_objects(corners_by_id, obj_aruco, h_img_to_grid)
            send_detected_objects(corners_by_id,
                                  obj_aruco, h_img_to_grid, sender)

        draw_status(frame, corners_by_id, obj_aruco, q_data, h_img_to_grid)

        cv2.imshow(config.WINDOW_CAMERA, frame)

        if aerial is not None:
            cv2.imshow(config.WINDOW_AERIAL, aerial)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Quitter avec 'q'.

    # shut down proprement les ressources.
    sender.disconnect()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
