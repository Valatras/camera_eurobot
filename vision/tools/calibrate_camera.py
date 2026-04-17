"""Capture et calcul des parametres intrinseques de la camera.

Usage :
    python tools/calibrate_camera.py [--output intrinsics.npz]

Controles en direct :
    [espace] capturer la pose courante (mire entiere visible)
    [c]      lancer la calibration (min 10 captures)
    [r]      reset (efface captures)
    [q]      quitter

Necessite une mire echiquier physique. Parametres par defaut :
    - 9x6 coins internes (= damier 10x7 cases)
    - case de 30 mm
Adapter via --cols, --rows, --size_mm.

Le fichier .npz genere est charge au demarrage par marker_detection.intrinsics.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Permet d'importer marker_detection depuis tools/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from marker_detection import config  # noqa: E402
from marker_detection.runtime import create_capture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibration intrinseque camera via mire echiquier")
    parser.add_argument("--cols", type=int, default=9,
                        help="Coins internes horizontaux (defaut 9)")
    parser.add_argument("--rows", type=int, default=6,
                        help="Coins internes verticaux (defaut 6)")
    parser.add_argument("--size-mm", type=float, default=30.0,
                        help="Taille d'une case en mm (defaut 30)")
    parser.add_argument("--output", default=os.path.join(HERE, "..", "intrinsics.npz"),
                        help="Fichier .npz de sortie")
    parser.add_argument("--min-captures", type=int, default=10,
                        help="Nombre min de captures avant calibration")
    args = parser.parse_args()

    pattern = (args.cols, args.rows)
    # Points 3D de reference (Z=0), exprimes en mm puis normalises en m.
    objp = np.zeros((args.cols * args.rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.size_mm / 1000.0  # en metres

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    last_capture_t = 0.0

    try:
        cap = create_capture()
    except RuntimeError as exc:
        print(f"[ERREUR] {exc}")
        return 1

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 1280, 720)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"Mire: {args.cols}x{args.rows} coins, case={args.size_mm} mm")
    print("[espace]=capturer  [c]=calibrer  [r]=reset  [q]=quitter")

    img_size: tuple[int, int] | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
        )

        preview = frame.copy()
        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(
                preview, pattern, corners_refined, found)

        status = f"captures: {len(objpoints)}"
        if len(objpoints) >= args.min_captures:
            status += " (OK, pressez [c] pour calibrer)"
        cv2.putText(preview, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0) if found else (0, 0, 255), 2)

        cv2.imshow("Calibration", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" ") and found and (time.time() - last_capture_t) > 0.5:
            objpoints.append(objp.copy())
            imgpoints.append(corners_refined)
            last_capture_t = time.time()
            print(f"[capture {len(objpoints)}] OK")
        elif key == ord("r"):
            objpoints.clear()
            imgpoints.clear()
            print("[reset]")
        elif key == ord("c"):
            if len(objpoints) < args.min_captures:
                print(
                    f"[WARN] Besoin d'au moins {args.min_captures} captures "
                    f"(actuel: {len(objpoints)})"
                )
                continue
            print("[calibration] en cours...")
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None,
            )
            # Erreur de reprojection moyenne en pixels.
            total_err = 0.0
            for i in range(len(objpoints)):
                proj, _ = cv2.projectPoints(
                    objpoints[i], rvecs[i], tvecs[i], K, dist)
                err = cv2.norm(imgpoints[i], proj,
                               cv2.NORM_L2) / len(proj)
                total_err += err
            mean_err = total_err / len(objpoints)
            print(f"[OK] RMS reprojection = {mean_err:.3f} px")
            print(f"K =\n{K}")
            print(f"dist = {dist.ravel()}")

            np.savez(
                args.output,
                camera_matrix=K,
                dist_coeffs=dist,
                image_width=img_size[0],
                image_height=img_size[1],
                reprojection_error_px=mean_err,
            )
            print(f"[SAUVEGARDE] {args.output}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
