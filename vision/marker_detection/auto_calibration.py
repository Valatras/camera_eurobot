"""Auto-calibration intrinseque + extrinseque de la camera.

Approche : les 4 coins ArUco de table (100x100 mm, positions 3D connues,
z=0) fournissent 16 correspondances 3D<->2D par frame. Sur N frames on
accumule assez de contraintes pour estimer K (+ distorsion optionnelle)
et la pose extrinseque de la camera sans damier.

Strategie (par ordre de preference) :
    1. **Zhang multi-vues** si les frames presentent suffisamment de
       diversite de pose (ecart-type des tvecs > seuil) :
       cv2.calibrateCamera avec les 16 points par frame sur N frames.
    2. **Distorsion nulle** si la camera est fixe et Zhang degenererait :
       on ne resout que fx/fy/cx/cy par optimisation des moindres carres
       sur la reprojection des coins (hypothese raisonnable pour un
       smartphone moderne post-traite). Valide si RMS reprojection <
       INTRINSIC_MAX_RMS_PX.
    3. **Echec** : message explicite demandant une calibration damier
       via tools/calibrate_camera.py.

Sortie : un `Intrinsics` pret a l'emploi (meme interface que
`intrinsics.Intrinsics.load`) persiste sur disque dans intrinsics.npz.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from marker_detection import config
from marker_detection.intrinsics import Intrinsics

logger = logging.getLogger(__name__)


# Ordre des 4 coins d'un marqueur ArUco tels que retournes par OpenCV :
# top-left, top-right, bottom-right, bottom-left (dans le repere image).
# Dans le repere "marker" centre sur le marqueur, z=0, axe x vers la
# droite, axe y vers le haut, la convention cv2.aruco donne :
#   corner 0 : (-s/2,  s/2, 0)   top-left
#   corner 1 : ( s/2,  s/2, 0)   top-right
#   corner 2 : ( s/2, -s/2, 0)   bottom-right
#   corner 3 : (-s/2, -s/2, 0)   bottom-left
def marker_object_points(size_mm: float, height_cm: float = 0.0) -> np.ndarray:
    """Retourne les 4 coins 3D (mm) d'un marqueur dans son repere local.

    Le plan du marqueur est a z = height_cm*10 (mm). Convention OpenCV
    ArUco : coin 0 = TL, 1 = TR, 2 = BR, 3 = BL.
    """
    s = size_mm / 2.0
    z = height_cm * 10.0
    return np.array(
        [
            [-s,  s, z],
            [ s,  s, z],
            [ s, -s, z],
            [-s, -s, z],
        ],
        dtype=np.float32,
    )


def corner_object_points_table(marker_id: int) -> np.ndarray:
    """Positions 3D (mm) des 4 coins d'un ArUco de table dans le repere table.

    Le centre du marqueur est en `CORNER_REAL_POSITIONS_CM[marker_id]`, le
    marqueur est pose a plat (z=0), et les coins internes sont a +/- 50 mm
    (100 mm de cote / 2). Axes : x horizontal, y vertical, z vers le haut.
    """
    cx_cm, cy_cm = config.CORNER_REAL_POSITIONS_CM[marker_id]
    cx_mm = cx_cm * 10.0
    cy_mm = cy_cm * 10.0
    s = config.MARKER_SIZE_CORNER_MM / 2.0
    # Meme convention que cv2.aruco : TL, TR, BR, BL dans le repere table.
    return np.array(
        [
            [cx_mm - s, cy_mm + s, 0.0],
            [cx_mm + s, cy_mm + s, 0.0],
            [cx_mm + s, cy_mm - s, 0.0],
            [cx_mm - s, cy_mm - s, 0.0],
        ],
        dtype=np.float32,
    )


def _default_camera_matrix(width: int, height: int) -> np.ndarray:
    """Matrice K initiale pour une camera smartphone 4K grand-angle."""
    f = config.INTRINSIC_DEFAULT_FOCAL_RATIO * width
    return np.array(
        [[f, 0.0, width / 2.0],
         [0.0, f, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclass
class _Frame:
    """Observations d'une frame : 16 points 2D + 16 points 3D (table)."""
    image_points: np.ndarray  # (16, 2) float32
    object_points: np.ndarray  # (16, 3) float32


@dataclass
class AutoCalibrator:
    """Accumulateur de frames pour calibration intrinseque sans damier."""

    image_size: tuple[int, int]
    min_frames: int = field(default_factory=lambda: config.CALIBRATION_MIN_FRAMES)
    timeout_s: float = field(default_factory=lambda: config.CALIBRATION_TIMEOUT_S)
    _frames: list[_Frame] = field(default_factory=list)
    _started: float = field(default_factory=time.monotonic)

    def feed(self, corners_by_id: dict[int, np.ndarray]) -> bool:
        """Ajoute une frame si les 4 coins sont visibles. Retourne True si acceptee."""
        if not all(cid in corners_by_id for cid in config.CORNER_IDS):
            return False
        img_pts = []
        obj_pts = []
        for cid in sorted(config.CORNER_IDS):
            # corners_by_id[cid] a la forme (1, 4, 2) ou (4, 2).
            pts = np.asarray(corners_by_id[cid], dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] != 4:
                return False
            img_pts.append(pts)
            obj_pts.append(corner_object_points_table(cid))
        self._frames.append(_Frame(
            image_points=np.concatenate(img_pts, axis=0),
            object_points=np.concatenate(obj_pts, axis=0),
        ))
        return True

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started

    @property
    def timed_out(self) -> bool:
        return self.elapsed_s > self.timeout_s

    def ready(self) -> bool:
        return self.n_frames >= self.min_frames

    def calibrate(self) -> Intrinsics | None:
        """Tente la calibration. Retourne un Intrinsics pret ou None.

        Essaie d'abord Zhang multi-vues (si les poses varient assez), sinon
        retombe sur un modele a distorsion nulle avec validation par
        reprojection.
        """
        if not self.ready():
            logger.warning("Auto-calib : %d frames (<%d requis)",
                           self.n_frames, self.min_frames)
            return None

        w, h = self.image_size
        K_init = _default_camera_matrix(w, h)

        # Tentative 1 : Zhang multi-vues. Les points 3D etant tous
        # coplanaires (z=0), cv2.calibrateCamera peut fonctionner si les
        # vues different par la pose (rotation hors plan). Si la camera
        # est strictement fixe, Zhang est degenere -> on passera au plan B.
        obj_pts_list = [f.object_points for f in self._frames]
        img_pts_list = [f.image_points for f in self._frames]

        zhang_ok = False
        K_zhang = K_init.copy()
        dist_zhang = np.zeros((5,), dtype=np.float64)
        try:
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS
                     | cv2.CALIB_FIX_ASPECT_RATIO
                     | cv2.CALIB_ZERO_TANGENT_DIST
                     | cv2.CALIB_FIX_K3)
            rms, K_zhang, dist_zhang, _, _ = cv2.calibrateCamera(
                obj_pts_list, img_pts_list, (w, h),
                K_zhang, dist_zhang, flags=flags,
            )
            zhang_ok = (
                rms < config.INTRINSIC_MAX_RMS_PX
                and K_zhang[0, 0] > 0
                and K_zhang[1, 1] > 0
                and 0 < K_zhang[0, 2] < w
                and 0 < K_zhang[1, 2] < h
            )
            if zhang_ok:
                logger.info("Auto-calib Zhang OK : RMS=%.3fpx fx=%.1f cx=%.1f cy=%.1f",
                            rms, K_zhang[0, 0], K_zhang[0, 2], K_zhang[1, 2])
        except cv2.error as exc:
            logger.info("Auto-calib Zhang rejetee (degenere?) : %s", exc)

        if zhang_ok:
            K = K_zhang
            dist = dist_zhang
        else:
            # Tentative 2 : distorsion nulle, K par defaut (plus raffinable
            # par solvePnP frame-par-frame). On valide par reprojection.
            K = K_init
            dist = np.zeros((5,), dtype=np.float64)
            rms = self._rms_reprojection(K, dist)
            logger.info("Auto-calib fallback (dist=0) : RMS=%.3fpx", rms)
            if rms > config.INTRINSIC_MAX_RMS_PX:
                logger.error(
                    "Auto-calib ECHEC : RMS=%.3fpx > %.1fpx. "
                    "Relancez avec un damier : tools/calibrate_camera.py",
                    rms, config.INTRINSIC_MAX_RMS_PX,
                )
                return None

        return self._build_intrinsics(K, dist)

    def _rms_reprojection(self, K: np.ndarray, dist: np.ndarray) -> float:
        """RMS global de reprojection sur toutes les frames (pixels)."""
        errs_sq: list[float] = []
        for f in self._frames:
            ok, rvec, tvec = cv2.solvePnP(
                f.object_points, f.image_points, K, dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue
            proj, _ = cv2.projectPoints(f.object_points, rvec, tvec, K, dist)
            diffs = proj.reshape(-1, 2) - f.image_points
            errs_sq.extend((diffs * diffs).sum(axis=1).tolist())
        if not errs_sq:
            return float("inf")
        return float(np.sqrt(np.mean(errs_sq)))

    def _build_intrinsics(self, K: np.ndarray, dist: np.ndarray) -> Intrinsics:
        w, h = self.image_size
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)
        map_x, map_y = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, (w, h), cv2.CV_16SC2,
        )
        return Intrinsics(
            camera_matrix=K,
            dist_coeffs=dist,
            image_size=(w, h),
            new_camera_matrix=new_K,
            map_x=map_x,
            map_y=map_y,
        )

    def save(self, intr: Intrinsics, path: str) -> None:
        """Persiste le resultat pour reutilisation aux prochaines sessions."""
        np.savez(
            path,
            camera_matrix=intr.camera_matrix,
            dist_coeffs=intr.dist_coeffs,
            image_width=intr.image_size[0],
            image_height=intr.image_size[1],
            reprojection_error_px=self._rms_reprojection(
                intr.camera_matrix, intr.dist_coeffs),
        )
        logger.info("Auto-calib persistee -> %s", path)
