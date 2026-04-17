"""Calibration intrinseque de la camera et correction de distorsion.

Utilise un fichier .npz genere par tools/calibrate_camera.py contenant la
matrice K (3x3) et les coefficients de distorsion (5+). Si le fichier n'est
pas present ou invalide, undistort_frame devient un no-op : le reste du
pipeline continue de fonctionner avec une precision degradee aux bords.

Usage:
    intr = Intrinsics.load("intrinsics.npz")
    if intr is not None:
        frame = intr.undistort(frame)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


DEFAULT_INTRINSICS_PATH = os.environ.get(
    "VISION_INTRINSICS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "intrinsics.npz"),
)


@dataclass
class Intrinsics:
    """Parametres intrinseques + cartes de remap pre-calculees."""

    camera_matrix: np.ndarray       # K 3x3
    dist_coeffs: np.ndarray         # (k1, k2, p1, p2, k3, ...)
    image_size: tuple[int, int]     # (w, h) utilise pour calibrer
    new_camera_matrix: np.ndarray   # K' apres getOptimalNewCameraMatrix
    map_x: np.ndarray
    map_y: np.ndarray

    @classmethod
    def load(cls, path: str = DEFAULT_INTRINSICS_PATH) -> "Intrinsics | None":
        """Charge un fichier .npz et prepare les cartes de remap.

        Retourne None si le fichier n'existe pas ou est invalide.
        """
        if not os.path.isfile(path):
            return None

        try:
            data = np.load(path)
            K = np.asarray(data["camera_matrix"], dtype=np.float64)
            dist = np.asarray(data["dist_coeffs"], dtype=np.float64)
            w = int(data["image_width"])
            h = int(data["image_height"])
        except (KeyError, ValueError, OSError) as exc:
            print(f"[WARN] Intrinsics invalides ({path}): {exc}")
            return None

        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)
        map_x, map_y = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, (w, h), cv2.CV_16SC2,
        )
        print(f"[OK] Intrinsics charges depuis {path} ({w}x{h})")
        return cls(
            camera_matrix=K,
            dist_coeffs=dist,
            image_size=(w, h),
            new_camera_matrix=new_K,
            map_x=map_x,
            map_y=map_y,
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Applique la correction de distorsion (remap precalcule, rapide)."""
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)
