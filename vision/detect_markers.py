"""Point d'entree du pipeline de detection ArUco."""

from __future__ import annotations

import argparse
import threading
import time

import cv2
import numpy as np

from marker_detection import config
from marker_detection.auto_calibration import AutoCalibrator
from marker_detection.detection import detect_aruco
from marker_detection.geometry import build_transforms, estimate_camera_pose
from marker_detection.intrinsics import DEFAULT_INTRINSICS_PATH, Intrinsics
from marker_detection.markers import build_detected_list, separate_markers
from marker_detection.pickup_zones import (
    PickupZoneTracker,
    count_nuts_in_garde_mangers,
)
from marker_detection.pose_smoother import PoseSmoother
from marker_detection.runtime import (
    create_aruco_detector,
    create_capture,
    create_clahe,
    create_windows,
)
from marker_detection.visualization import (
    compute_aerial,
    draw_aerial_overlay,
    draw_fallback_overlay,
)
from marker_detection.esp32_sender import ESP32Sender
from marker_detection.dashboard_sender import DashboardSender
from marker_detection.mjpeg_server import MjpegServer


class CaptureThread:
    """Thread dedie a la capture : ne conserve que la derniere frame."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2)


class ReplayCapture:
    """Pseudo-capture qui lit une serie d'images depuis un dossier.

    Utile pour rejouer une scene enregistree (diagnostic, tests de
    non-regression, debug sans camera). Respecte l'interface de
    ``CaptureThread`` : ``read()`` + ``stop()``.
    """

    def __init__(self, directory: str, fps: float = 15.0, loop: bool = True) -> None:
        import os
        import glob
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files: list[str] = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(directory, p)))
            files.extend(glob.glob(os.path.join(directory, p.upper())))
        self._files = sorted(set(files))
        if not self._files:
            raise RuntimeError(f"Aucune image trouvee dans {directory}")
        self._idx = 0
        self._period = 1.0 / max(1e-3, fps)
        self._last_read = 0.0
        self._loop = loop
        self._done = False

    def read(self) -> np.ndarray | None:
        now = time.perf_counter()
        if now - self._last_read < self._period:
            return None
        if self._done:
            return None
        path = self._files[self._idx]
        frame = cv2.imread(path)
        self._last_read = now
        self._idx += 1
        if self._idx >= len(self._files):
            if self._loop:
                self._idx = 0
            else:
                self._done = True
        return frame

    def stop(self) -> None:
        self._done = True


def _run_auto_calibration(
    capture: "CaptureThread | ReplayCapture",
    detector,
    clahe,
) -> Intrinsics | None:
    """Phase de warm-up : accumule des frames et calibre sans damier.

    Retourne un Intrinsics valide et persiste intrinsics.npz, ou None en
    cas d'echec (timeout ou RMS trop eleve).
    """
    # Attendre la premiere frame pour connaitre la taille d'image.
    t_wait = time.monotonic()
    first_frame = None
    while first_frame is None:
        first_frame = capture.read()
        if time.monotonic() - t_wait > 10.0:
            print("[CALIB] Timeout : aucune frame recue de la capture.")
            return None
        time.sleep(0.05)
    h, w = first_frame.shape[:2]
    print(f"[CALIB] Auto-calibration demarree ({w}x{h})")

    calib = AutoCalibrator(image_size=(w, h))
    last_progress = 0
    while not calib.timed_out:
        frame = capture.read()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        a_corners, a_ids = detect_aruco(gray, detector, clahe)
        corners_by_id: dict[int, np.ndarray] = {}
        for marker_id, corner in zip(a_ids, a_corners):
            if marker_id in config.CORNER_IDS:
                corners_by_id[marker_id] = corner
        calib.feed(corners_by_id)
        if calib.n_frames > last_progress:
            last_progress = calib.n_frames
            print(f"[CALIB] {calib.n_frames}/{calib.min_frames} frames acceptees "
                  f"(t={calib.elapsed_s:.1f}s)")
        if calib.ready():
            intr = calib.calibrate()
            if intr is not None:
                try:
                    calib.save(intr, DEFAULT_INTRINSICS_PATH)
                    print(f"[CALIB] Succes -> {DEFAULT_INTRINSICS_PATH}")
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[CALIB] Impossible de sauver intrinsics.npz : {exc}")
                return intr
    print(f"[CALIB] Timeout ({calib.timeout_s:.1f}s) : "
          f"{calib.n_frames} frames, {len(config.CORNER_IDS)} coins requis. "
          f"Verifier que les 4 coins ArUco sont visibles.")
    return None


def main() -> None:
    """Boucle principale du pipeline."""
    parser = argparse.ArgumentParser(description="Pipeline de detection ArUco")
    parser.add_argument("--gui", action="store_true",
                        help="Activer l'affichage GUI (fenetres OpenCV)")
    parser.add_argument("--profile", action="store_true",
                        help="Afficher les timings par frame")
    parser.add_argument("--no-stream", action="store_true",
                        help="Desactiver le serveur MJPEG")
    parser.add_argument("--clahe-clip", type=float, default=config.CLAHE_CLIP_LIMIT,
                        help=f"CLAHE clip limit (defaut: {config.CLAHE_CLIP_LIMIT})")
    parser.add_argument("--clahe-tile", type=int, default=config.CLAHE_TILE_GRID_SIZE[0],
                        help=f"CLAHE tile size (defaut: {config.CLAHE_TILE_GRID_SIZE[0]})")
    parser.add_argument("--replay-dir", type=str, default=None,
                        help="Rejouer un dossier d'images au lieu d'utiliser la camera")
    parser.add_argument("--replay-fps", type=float, default=15.0,
                        help="FPS cible en mode replay (defaut: 15)")
    parser.add_argument("--replay-once", action="store_true",
                        help="Arreter a la fin du dossier (pas de boucle)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Force l'auto-calibration intrinseque au demarrage"
                             " (ignore intrinsics.npz existant)")
    parser.add_argument("--no-auto-calib", action="store_true",
                        help="Desactive l'auto-calibration sans damier (utilise"
                             " uniquement intrinsics.npz ou echoue)")
    args = parser.parse_args()

    try:
        if args.replay_dir:
            cap = None
            print(
                f"[REPLAY] Lecture de {args.replay_dir} @ {args.replay_fps} fps")
        else:
            cap = create_capture()
        sender = ESP32Sender()
        sender.connect()
    except RuntimeError as exc:
        print(f"[ERREUR] {exc}")
        return

    dashboard_sender = None
    if config.DASHBOARD_URL:
        dashboard_sender = DashboardSender(config.DASHBOARD_URL)
        if dashboard_sender.connect():
            print(f"[OK] Dashboard connecte: {config.DASHBOARD_URL}")
        else:
            print(f"[WARN] Dashboard non accessible: {config.DASHBOARD_URL}")
            print("       Verifier que le serveur tourne (pnpm dev dans dashboard/)")

    gui_active = create_windows(headless=not args.gui)

    # MJPEG streaming server (for dashboard video feed).
    mjpeg: MjpegServer | None = None
    if not args.no_stream:
        mjpeg = MjpegServer(config.MJPEG_PORT)
        mjpeg.start()
        print(
            f"[OK] MJPEG stream: http://localhost:{config.MJPEG_PORT}/stream")

    detector = create_aruco_detector()
    # Utiliser CLAHE personnalisé si spécifié, sinon config par défaut
    if args.clahe_clip != config.CLAHE_CLIP_LIMIT or args.clahe_tile != config.CLAHE_TILE_GRID_SIZE[0]:
        clahe = cv2.createCLAHE(
            clipLimit=args.clahe_clip,
            tileGridSize=(args.clahe_tile, args.clahe_tile),
        )
        print(
            f"[OK] CLAHE personnalise: clip={args.clahe_clip}, tile={args.clahe_tile}x{args.clahe_tile}")
    else:
        clahe = create_clahe()

    print("[OK] Detection ArUco optimisee pour petits marqueurs lointains (pleine resolution)")

    # Chargement des parametres intrinseques. La calibration est
    # OBLIGATOIRE pour atteindre une precision sous-cm (homographie 2D
    # seule sature a ~1-2 cm aux bords). Flux :
    #   1. Tenter de charger intrinsics.npz existant (sauf --recalibrate).
    #   2. Sinon, auto-calibrer sans damier depuis les 4 coins ArUco.
    #   3. En dernier recours (echec), demander calibrate_camera.py damier.
    intrinsics: Intrinsics | None = None
    if not args.recalibrate:
        intrinsics = Intrinsics.load()
    if intrinsics is None:
        print("[CALIB] intrinsics.npz indisponible ; auto-calibration demarre...")

    # Cache homographie (les 4 coins sont fixes sur la table).
    cached_transforms: tuple | None = None
    cached_transforms_ts: float = 0.0
    corners_stable_count = 0
    # Phase 12.5 : duci pour eviter le figeage sur homographie bruitee.
    CORNERS_STABLE_THRESHOLD = 10

    pose_smoother = PoseSmoother()
    zone_tracker = PickupZoneTracker()

    capture = CaptureThread(cap) if cap is not None else ReplayCapture(
        args.replay_dir, fps=args.replay_fps, loop=not args.replay_once,
    )

    # ------------------------------------------------------------------
    # Phase warm-up : auto-calibration intrinseque + extrinseque.
    # Bloque au demarrage jusqu'a convergence. En cas d'echec, sortie 1.
    # ------------------------------------------------------------------
    if intrinsics is None and not args.no_auto_calib:
        intrinsics = _run_auto_calibration(
            capture, detector, clahe,
        )
        if intrinsics is None:
            print("[FATAL] Auto-calibration echouee. Lancer :")
            print("        python tools/calibrate_camera.py")
            print("        puis redemarrer detect_markers.py")
            capture.stop()
            return
    if intrinsics is None:
        print("[FATAL] Pas d'intrinsics.npz et --no-auto-calib specifie.")
        capture.stop()
        return

    # Pose extrinseque de la camera (dans le repere table, mm). Mise a
    # jour a chaque frame ou les 4 coins sont vus, lissee par EMA.
    camera_rvec: np.ndarray | None = None
    camera_tvec: np.ndarray | None = None
    extrinsic_logged = False

    frame_count = 0
    fps_t0 = time.perf_counter()

    while True:
        frame = capture.read()
        if frame is None:
            continue

        t0 = time.perf_counter() if args.profile else 0

        # Correction de distorsion (si calibration intrinseque disponible).
        if intrinsics is not None:
            frame = intrinsics.undistort(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t1 = time.perf_counter() if args.profile else 0

        a_corners, a_ids = detect_aruco(
            gray, detector, clahe)  # Pleine résolution pour meilleurs résultats sur petits marqueurs

        t2 = time.perf_counter() if args.profile else 0

        # Utiliser les detections brutes comme dans tune_detection.py.
        # Le reste du pipeline (homographie, envoi, overlay) part de cette base.
        corners_by_id, obj_aruco = separate_markers(a_ids, a_corners)

        # Homographie : utiliser le cache si les coins n'ont pas bouge.
        n_corners_detected = sum(
            1 for mid in config.CORNER_IDS if mid in corners_by_id)
        now_ts = time.monotonic()
        homography_stale = False
        if n_corners_detected >= 3:
            corners_stable_count += 1
            if corners_stable_count >= CORNERS_STABLE_THRESHOLD and cached_transforms is not None:
                # Reutiliser le cache : coins stables detectes, pas besoin de
                # recalculer l'homographie.
                h_img_to_grid, h_grid_to_img, h_img_to_aerial, h_img_to_cm, table_pts, aruco_pts, _ = cached_transforms
            else:
                transforms = build_transforms(corners_by_id)
                h_img_to_grid, h_grid_to_img, h_img_to_aerial, h_img_to_cm, table_pts, aruco_pts, _ = transforms
                # Ne cacher que si la calibration est complete et valide.
                if h_img_to_grid is not None and n_corners_detected == 4:
                    cached_transforms = transforms
                    cached_transforms_ts = now_ts
                elif h_img_to_cm is None and cached_transforms is not None:
                    # Rejet par validation RMS : retomber sur le cache recent
                    # plutot que de perdre la frame entiere.
                    age = now_ts - cached_transforms_ts
                    if age <= config.HOMOGRAPHY_CACHE_TTL_S:
                        h_img_to_grid, h_grid_to_img, h_img_to_aerial, h_img_to_cm, table_pts, aruco_pts, _ = cached_transforms
                        homography_stale = True
        else:
            corners_stable_count = 0
            # Cache avec TTL : au-dela de HOMOGRAPHY_CACHE_TTL_S la camera
            # peut avoir bouge -> flag stale pour que Dave sache que la
            # donnee a du mou.
            if cached_transforms is not None:
                age = now_ts - cached_transforms_ts
                if age > config.HOMOGRAPHY_CACHE_TTL_S:
                    cached_transforms = None
                    h_img_to_grid = h_grid_to_img = h_img_to_aerial = h_img_to_cm = table_pts = aruco_pts = None
                    print(
                        f"[WARN] Cache homographie perime ({age:.1f}s > "
                        f"{config.HOMOGRAPHY_CACHE_TTL_S:.1f}s), camera indisponible"
                    )
                else:
                    h_img_to_grid, h_grid_to_img, h_img_to_aerial, h_img_to_cm, table_pts, aruco_pts, _ = cached_transforms
                    homography_stale = True
            else:
                h_img_to_grid = h_grid_to_img = h_img_to_aerial = h_img_to_cm = table_pts = aruco_pts = None

        t3 = time.perf_counter() if args.profile else 0

        # Envoi des donnees a chaque frame.
        corners_ok = n_corners_detected >= 3

        # Rafraichir la pose extrinseque de la camera (solvePnP sur les
        # 4 coins) chaque fois que >= 2 coins sont visibles. Valeurs
        # lissees par EMA implicite via remplacement progressif.
        if n_corners_detected >= 2:
            pose = estimate_camera_pose(
                corners_by_id,
                intrinsics.camera_matrix,
                intrinsics.dist_coeffs,
            )
            if pose is not None:
                new_rvec, new_tvec = pose
                if camera_rvec is None:
                    camera_rvec, camera_tvec = new_rvec, new_tvec
                else:
                    camera_rvec = 0.7 * camera_rvec + 0.3 * new_rvec
                    camera_tvec = 0.7 * camera_tvec + 0.3 * new_tvec
                if not extrinsic_logged:
                    from marker_detection.geometry import camera_extrinsics_summary
                    summary = camera_extrinsics_summary(
                        camera_rvec, camera_tvec)
                    print(f"[CALIB] Pose camera mesuree : "
                          f"x={summary['cam_x_cm']:.1f} cm, "
                          f"y={summary['cam_y_cm']:.1f} cm, "
                          f"hauteur={summary['cam_height_cm']:.1f} cm")
                    extrinsic_logged = True

        if h_img_to_cm is not None:
            detected = build_detected_list(
                corners_by_id, obj_aruco, h_img_to_cm,
                K=intrinsics.camera_matrix,
                dist=intrinsics.dist_coeffs,
                camera_rvec=camera_rvec,
                camera_tvec=camera_tvec,
            )
            dashboard_detected = build_detected_list(
                corners_by_id, obj_aruco, h_img_to_cm,
                prefer_pnp=True,
            )
            # Lissage EMA sur (x, y, angle) en coord table pour reduire le
            # jitter issu de la detection pixel (~2 cm sur les bords).
            detected = pose_smoother.smooth(detected)
            # Agregation noisettes par zone de ramassage (historique temporel).
            zone_states = zone_tracker.update(detected)
            gm_counts = count_nuts_in_garde_mangers(detected)
            zone_sequences = {zid: st.sequence()
                              for zid, st in zone_states.items()}
            sender.send_vision_state(
                detected=detected,
                zone_sequences=zone_sequences,
                gm_counts=gm_counts,
                stale=homography_stale,
                corners_ok=corners_ok,
            )
        else:
            detected = []
            dashboard_detected = []
            zone_sequences = zone_tracker.snapshot()
            gm_counts = {}
            pose_smoother.reset()
        if dashboard_sender is not None:
            dashboard_sender.send_detections(
                dashboard_detected,
                corners_ok and not homography_stale,
                zone_sequences=zone_sequences,
                gm_counts=gm_counts,
                stale=homography_stale,
            )

        t4 = time.perf_counter() if args.profile else 0

        # Annotation : vue aerienne si calibree, sinon perspective avec overlay.
        need_display = gui_active or mjpeg is not None
        if need_display:
            aerial = compute_aerial(frame, h_img_to_aerial)

            if aerial is not None:
                draw_aerial_overlay(aerial, corners_by_id, obj_aruco,
                                    h_img_to_aerial, h_img_to_cm, n_corners_detected)
                stream_frame = aerial
            else:
                display = cv2.resize(frame, (config.DISPLAY_W, config.DISPLAY_H),
                                     interpolation=cv2.INTER_AREA)
                scale = config.DISPLAY_W / config.FRAME_W
                scaled_corners = {mid: c * scale for mid,
                                  c in corners_by_id.items()}
                scaled_obj = [(mid, c * scale) for mid, c in obj_aruco]
                draw_fallback_overlay(display, scaled_corners, scaled_obj)
                stream_frame = display

            # Push frame to MJPEG server.
            if mjpeg is not None:
                _, jpeg = cv2.imencode(".jpg", stream_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY])
                mjpeg.set_frame(jpeg.tobytes())

            # GUI window.
            if gui_active:
                cv2.imshow(config.WINDOW_CAMERA, stream_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        t5 = time.perf_counter() if args.profile else 0

        if args.profile and frame_count % 10 == 0:
            elapsed = time.perf_counter() - fps_t0
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[{fps:.1f} fps] gray={1000*(t1-t0):.0f}ms "
                  f"detect={1000*(t2-t1):.0f}ms "
                  f"transform={1000*(t3-t2):.0f}ms "
                  f"send={1000*(t4-t3):.0f}ms "
                  f"display={1000*(t5-t4):.0f}ms "
                  f"total={1000*(t5-t0):.0f}ms")

        frame_count += 1

    capture.stop()
    if mjpeg is not None:
        mjpeg.stop()
    sender.disconnect()
    if dashboard_sender is not None:
        dashboard_sender.disconnect()
    cap.release()
    if gui_active:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
