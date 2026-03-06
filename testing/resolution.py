import os
from pathlib import Path

# Keep HighGUI on X11/XWayland and point to a system font folder when available.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
if Path("/usr/share/fonts/truetype/dejavu").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2

import config


PREFERRED_CAMERA_NAME = "Iriun Webcam"
PROBE_RESOLUTIONS = [
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1280, 720),
]


def get_camera_name(camera_index: int) -> str:
    name_path = Path(f"/sys/class/video4linux/video{camera_index}/name")
    if name_path.exists():
        return name_path.read_text(encoding="utf-8").strip()
    return "Nom inconnu"


def build_candidate_indices(
    preferred_index: int, preferred_name: str | None, max_index: int
) -> list[int]:
    existing_indices = [
        idx for idx in range(max_index) if Path(f"/sys/class/video4linux/video{idx}").exists()
    ]
    if not existing_indices:
        return []

    name_matches = []
    others = []
    for idx in existing_indices:
        name = get_camera_name(idx)
        if preferred_name and preferred_name.lower() in name.lower():
            name_matches.append(idx)
        else:
            others.append(idx)

    ordered = []
    if preferred_index in existing_indices:
        preferred_name_at_index = get_camera_name(preferred_index)
        if (
            not preferred_name
            or preferred_name.lower() in preferred_name_at_index.lower()
        ):
            ordered.append(preferred_index)

    for idx in name_matches + others:
        if idx not in ordered:
            ordered.append(idx)

    return ordered


def open_camera_with_fallback(preferred_index: int, max_index: int = 5):
    tried_indices = []
    candidate_indices = build_candidate_indices(
        preferred_index, PREFERRED_CAMERA_NAME, max_index
    )

    for camera_index in candidate_indices:
        tried_indices.append(camera_index)
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap, camera_index, tried_indices
        cap.release()

    return None, None, tried_indices


def log_capture_state(cap: cv2.VideoCapture, prefix: str) -> None:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{prefix} -> CAP_PROP: {width}x{height} @ {fps:.2f} FPS")


def probe_supported_resolutions(cap: cv2.VideoCapture) -> None:
    print("\nProbe des résolutions (source de vérité: frame.shape):")
    for req_w, req_h in PROBE_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        prop_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        prop_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
        if ok:
            frame_h, frame_w = frame.shape[:2]
            print(
                f"- demandé {req_w}x{req_h} -> CAP_PROP {prop_w}x{prop_h}, frame {frame_w}x{frame_h}"
            )
        else:
            print(
                f"- demandé {req_w}x{req_h} -> CAP_PROP {prop_w}x{prop_h}, lecture impossible"
            )


def main() -> None:
    if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)

    cap, camera_index, tried_indices = open_camera_with_fallback(config.CAMERA_INDEX)
    if not cap or camera_index is None:
        print(
            f"Erreur : Aucune caméra trouvée. Index testés: {', '.join(map(str, tried_indices))}"
        )
        return

    camera_name = get_camera_name(camera_index)
    print(f"Caméra ouverte à l'index {camera_index} ({camera_name}) avec backend V4L2")
    if camera_index != config.CAMERA_INDEX:
        print(
            f"Note: index demandé {config.CAMERA_INDEX} indisponible, fallback sur {camera_index}"
        )

    log_capture_state(cap, "Avant configuration")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    log_capture_state(cap, "Après configuration")
    print(f"Demandé: {config.FRAME_W}x{config.FRAME_H} @ {config.CAMERA_FPS} FPS")

    ok, frame = cap.read()
    if not ok:
        print("Erreur : Impossible de lire une image après configuration.")
        cap.release()
        return

    frame_h, frame_w = frame.shape[:2]
    print(f"Résolution réelle (frame.shape): {frame_w}x{frame_h}")
    if (frame_w, frame_h) != (config.FRAME_W, config.FRAME_H):
        print(
            "Résolution demandée non atteinte : mode non exposé par le driver/app caméra."
        )

    probe_supported_resolutions(cap)

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 1920, 1080)
    print("\nAppuyez sur 'q' pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
