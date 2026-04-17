# Vision — Eurobot 2026

Pipeline de détection ArUco + QR par caméra surplombante pour la localisation des éléments de jeu (robots, noisettes, zones) sur le terrain.

**Stack** : Python 3.12 · OpenCV 4.13 · ArUco DICT_4X4_50 · NumPy · Socket.IO

---

## Table des matières

- [Matériel — Caméra surplombante](#matériel--caméra-surplombante)
- [Architecture](#architecture)
- [Modules](#modules)
- [Configuration](#configuration)
- [Communication vers ESP32](#communication-vers-esp32)
- [Communication vers Dashboard](#communication-vers-dashboard)
- [Marqueurs ArUco](#marqueurs-aruco)
- [Commandes](#commandes)

---

## Matériel — Caméra surplombante

Le pipeline nécessite une caméra USB surplombante (vue zénithale) installée au-dessus de la table Eurobot, dans la **zone déportée** (règle G.4). On utilise un smartphone comme webcam, connecté en USB au laptop (GPU recommandé pour le traitement en temps réel).

### Choix de l'application webcam (2026)

| Application  | Licence                    | 4K gratuit                           | USB             | Linux V4L2          | Remarques                                                                                                                         |
| ------------ | -------------------------- | ------------------------------------ | --------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **scrcpy**   | Apache 2.0 (open source)   | Oui (résolution native du téléphone) | Oui             | Oui (`--v4l2-sink`) | **Recommandé.** 138k★ GitHub, latence 35-70 ms, pas d'app à installer sur le téléphone. Requiert Android 12+ pour le mode caméra. |
| Iriun Webcam | Gratuit (closed source)    | Oui (watermark en free)              | Oui             | Oui                 | Très populaire (10 M+ downloads), v2.9.4 (avril 2026). Pro payant. Bon backup.                                                    |
| DroidCam     | Gratuit / DroidCamX payant | Non (SD en free, HD payant)          | Oui             | Oui (V4L2 + ALSA)   | Alternative solide. HD/4K nécessite DroidCamX.                                                                                    |
| IP Webcam    | Gratuit                    | Variable                             | Non (WiFi only) | Via MJPEG           | WiFi = latence trop élevée pour le temps réel.                                                                                    |

**Recommandation** : utiliser **scrcpy** — entièrement gratuit, open source, sans app à installer, latence USB très faible, pilote V4L2 natif sur Linux.

### Installation scrcpy (méthode recommandée)

```bash
# NixOS / Nix
nix-shell -p scrcpy android-tools

# Ubuntu / Debian
sudo apt install scrcpy adb

# Arch
sudo pacman -S scrcpy android-tools
```

### Lancer la caméra

1. Activer le **débogage USB** sur le téléphone (Paramètres → Options développeur).
2. Brancher le téléphone en USB et autoriser la connexion ADB.
3. Lancer scrcpy en mode caméra → V4L2 :

```bash
# Créer le device V4L2 si nécessaire
sudo modprobe v4l2loopback

# Lancer la caméra arrière en 1920x1080 sur /dev/video2
scrcpy --video-source=camera \
       --camera-size=1920x1080 \
       --camera-facing=back \
       --v4l2-sink=/dev/video2 \
       --no-playback

# Pour du 4K (si le téléphone le supporte) :
scrcpy --video-source=camera \
       --camera-size=3840x2160 \
       --camera-facing=back \
       --v4l2-sink=/dev/video2 \
       --no-playback
```

4. Le pipeline OpenCV lit ensuite `/dev/video2` via `CAMERA_INDEX = 2` dans `config.py`.

### Alternative : Iriun Webcam

1. Installer l'app Iriun Webcam sur le téléphone (Play Store / App Store).
2. Installer le pilote Linux : télécharger depuis [iriun.com](https://iriun.com) et suivre les instructions.
3. Brancher en USB, lancer Iriun sur le téléphone → la caméra apparaît comme `/dev/video0`.
4. Le pipeline OpenCV lit le device via `CAMERA_INDEX = 0` dans `config.py`.

---

## Architecture

```
Smartphone (caméra arrière, 4K @ 30fps)
    │  USB
    ▼
scrcpy --v4l2-sink → /dev/videoN
    │
    ▼
detect_markers.py                 ← Point d'entrée
    │
    ├── marker_detection/
    │   ├── config.py              ← Résolution, IDs, géométrie table, TEAM_COLOR
    │   ├── detection.py           ← ArUco + QR multi-résolution, CLAHE
    │   ├── geometry.py            ← Homographie image → cm, image → grille, → aerial
    │   ├── markers.py             ← Classification (NUT_BLUE/YELLOW/EMPTY, BR1-5, YR1-5)
    │   ├── tracking.py            ← Lissage temporel (buffer=3, min_hits=2)
    │   ├── visualization.py       ← Overlay : grille, marqueurs, vue aérienne
    │   ├── runtime.py             ← Factory (caméra, détecteurs, fenêtres)
    │   ├── esp32_sender.py        ← Série USB → ESP32 (Obstacle X Y)
    │   └── dashboard_sender.py    ← Socket.IO → Dashboard web
    │
    └── aruco/                     ← PDFs de référence des marqueurs
```

### Flux de données

```
detect_markers.py
    │
    ├──► ESP32Sender (série USB)
    │       Format : "Obstacle X Y\n" (adversaires uniquement, coords cm)
    │       → ESP32 Bridge → ESP-NOW → Robots
    │
    └──► DashboardSender (Socket.IO)
            Événement : visionDetections
            Payload : {nuts, robots, opponents, corners_ok, ts_ms}
            → Dashboard server → broadcast web + relai série
```

---

## Modules

| Module              | Fichier               | Rôle                                                                                                                                                                            |
| ------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Config**          | `config.py`           | Résolution 3840×2160, IDs ArUco (coins 20-23, noisettes 36/41/47), géométrie table (300×200 cm), `TEAM_COLOR`, `DASHBOARD_URL`                                                  |
| **Detection**       | `detection.py`        | Détection multi-résolution ArUco + QR. Preprocessing CLAHE (clip=2.5, tile 8×8). Validation par area min/max, aspect ratio et variance                                          |
| **Geometry**        | `geometry.py`         | Homographie image → cm (`to_cm`), image → grille (`to_cell`), image → aerial. Extrapolation des coins de table (inset 500 mm). `compute_angle` pour l'orientation des marqueurs |
| **Markers**         | `markers.py`          | Classification par ID : `NUT_BLUE` (36), `NUT_YELLOW` (47), `NUT_EMPTY` (41), `BR1-5`, `YR1-5`. Fonctions `is_opponent` / `is_ally` selon `TEAM_COLOR`. Sortie en cm            |
| **Tracking**        | `tracking.py`         | Lissage temporel : buffer de 3 frames, détection validée après 2 hits minimum. Élimine le flickering                                                                            |
| **Visualization**   | `visualization.py`    | Rendu overlay : grille, contours de table, marqueurs annotés, QR codes, vue aérienne projetée                                                                                   |
| **Runtime**         | `runtime.py`          | Fonctions factory : initialisation caméra, création des détecteurs ArUco/QR, gestion des fenêtres OpenCV                                                                        |
| **ESP32Sender**     | `esp32_sender.py`     | Envoi des **adversaires uniquement** via série USB. Format `Obstacle X Y\n` + `END\n`. Auto-détection du port par VID/PID                                                       |
| **DashboardSender** | `dashboard_sender.py` | Envoi de toutes les détections via Socket.IO : noisettes, robots alliés, adversaires. Reconnexion automatique                                                                   |

---

## Configuration

### Variables d'environnement

| Variable        | Défaut                  | Description                                                                     |
| --------------- | ----------------------- | ------------------------------------------------------------------------------- |
| `TEAM_COLOR`    | `blue`                  | Couleur de l'équipe (`blue` ou `yellow`). Détermine qui est adversaire vs allié |
| `DASHBOARD_URL` | `http://localhost:3001` | URL du serveur Socket.IO du dashboard                                           |

### Constantes dans `config.py`

| Paramètre                  | Valeur           | Description                                       |
| -------------------------- | ---------------- | ------------------------------------------------- |
| `FRAME_W × FRAME_H`        | 3840 × 2160      | Résolution caméra (4K)                            |
| `CAMERA_INDEX`             | 0                | Index du device V4L2 (adapter selon scrcpy/Iriun) |
| `CAMERA_FPS`               | 30               | Framerate cible                                   |
| `CORNER_IDS`               | {20, 21, 22, 23} | IDs ArUco pour les 4 coins de table               |
| `CORNER_ORDER`             | [23, 22, 20, 21] | Ordre : TL, TR, BR, BL                            |
| `NUT_BLUE_ID`              | 36               | ArUco de la caisse de noisettes bleue             |
| `NUT_YELLOW_ID`            | 47               | ArUco de la caisse de noisettes jaune             |
| `NUT_EMPTY_ID`             | 41               | ArUco de la caisse de noisettes vide              |
| `TABLE_W_CM × TABLE_H_CM`  | 300 × 200        | Dimensions de la table en cm                      |
| `ARUCO_INSET_MM`           | 600              | Distance des centres ArUco coin → bord table      |
| `GRID_COLS × GRID_ROWS`    | 30 × 20          | Grille de détection (**10 cm/cellule**)           |
| `CAMERA_HEIGHT_CM`         | 136              | Hauteur caméra au-dessus de la table (hypothèse)  |
| `CAMERA_TABLE_POSITION_CM` | (160, 200)       | Projection au sol du centre optique (hypothèse)   |
| `NUT_HEIGHT_CM`            | 3                | Hauteur d'une noisette (parallaxe)                |
| `ROBOT_HEIGHT_CM`          | 43               | Hauteur des robots (parallaxe)                    |

---

## Communication vers ESP32

Le module `esp32_sender.py` envoie **uniquement les positions adversaires** à l'ESP32 Bridge via USB série.

**Protocole actuel** (texte, ligne par ligne) :

```
STALE 0|1                   # homographie en cache obsolète
CORNERS_OK 0|1              # ≥3 coins visibles
OPP x_cm y_cm theta_dcdeg   # adversaire avec angle (deg × 10)
Obstacle X Y                # legacy, même info, rétrocompat
ZR id C1 C2 C3 C4 C5        # séquence couleur dans la zone id (B/Y/E/?)
GM id count                 # nombre de noisettes dans le garde-manger id
END                         # fin de trame
```

- **X, Y** : coordonnées en **cm** sur la table (0,0 = coin supérieur gauche)
- **theta_dcdeg** : angle en **décicentièmes de degré** (angle°×10), plage ±1800
- **Cx** : couleur du slot x dans la ZR — `B` (bleue), `Y` (jaune), `E` (vide), `?` (inconnue)
- Rétrocompat : les anciens firmwares n'ayant que le parser `Obstacle X Y` ignorent les autres lignes
- Côté firmware : les lignes sont transmises telles quelles par le bridge (ESP-NOW `MSG_CMD` texte) et interprétées par [`VisionState::applyLine()`](../firmware/lib/EurobotCore/src/VisionState.cpp). L'état est exposé via le singleton `g_visionState` (pose adversaire avec angle, séquences ZR, comptes GM, flag `stale`, flag `cornersOk`, fraîcheur). Aucun nouveau type `MSG_*` n'est introduit — le chemin de transport reste celui des commandes texte (`@all …` ou peer par défaut).
- **Auto-détection** du port série ESP32 par VID/PID USB (CP210x, CH340, FTDI, Espressif native)

---

## Communication vers Dashboard

Le module `dashboard_sender.py` envoie toutes les détections au dashboard web via **Socket.IO**.

**Événement** : `visionDetections`

**Payload** :

```json
{
  "nuts": [{ "label": "NUT_BLUE", "x": 150.3, "y": 80.1, "angle": 12.5 }],
  "robots": [{ "label": "BR1", "x": 42.0, "y": 100.5, "angle": -30.2 }],
  "opponents": [{ "label": "YR2", "x": 230.0, "y": 60.0, "angle": 90.0 }],
  "corners_ok": true,
  "ts_ms": 1719849600000
}
```

- Les robots/adversaires sont triés selon `TEAM_COLOR`
- `corners_ok` est vrai dès que ≥3 coins sont visibles (homographie affine ou perspective) — **ne distingue pas actuellement l'usage d'un cache obsolète** (cf. §Limitations)
- ⚠️ **Le relais dashboard → port série du bridge n'est pas actif en match** : le forward dans `dashboard/apps/server/.../socket.ts` est `TODO` (cf. [MATCH_LOGIC_AUDIT.md §2.5](../firmware/MATCH_LOGIC_AUDIT.md)). En état actuel, seul le chemin direct `detect_markers.py` → port USB du bridge alimente les robots en match

---

## Marqueurs ArUco

Dictionnaire : **DICT_4X4_250** (cf. [`runtime.py`](marker_detection/runtime.py)). Les PDFs imprimables sont dans `aruco/`.

| IDs            | Type                          | Rôle                                                        |
| -------------- | ----------------------------- | ----------------------------------------------------------- |
| 20, 21, 22, 23 | Coins de table                | Fixes, collés aux 4 coins du terrain. Base de l'homographie |
| 1 – 5          | Robots bleus (BR1-BR5)        | Marqueurs sur les robots de l'équipe bleue                  |
| 6 – 10         | Robots jaunes (YR1-YR5)       | Marqueurs sur les robots de l'équipe jaune                  |
| 36             | Noisettes bleues (NUT_BLUE)   | Caisse de noisettes de l'équipe bleue                       |
| 41             | Noisettes vides (NUT_EMPTY)   | Caisse de noisettes neutre                                  |
| 47             | Noisettes jaunes (NUT_YELLOW) | Caisse de noisettes de l'équipe jaune                       |

---

## Calibration & précision

### Modes de calibration d'homographie

| Coins ArUco visibles | Comportement                                                                  |
| -------------------- | ----------------------------------------------------------------------------- |
| **4 / 4**            | Homographie perspective complète (`cv2.getPerspectiveTransform`)              |
| **3 / 4**            | Transformation affine (`cv2.estimateAffine2D`) — coin manquant extrapolé      |
| **2 / 4**            | Pas de nouvelle calib → **réutilise le cache** si disponible (cf. ci-dessous) |
| **< 2**              | Cache réutilisé s'il existe, sinon caméra marquée indisponible                |

Un cache simple de la dernière homographie valide est conservé en RAM ([`detect_markers.py`](detect_markers.py)) avec un **TTL de 5 s** (`HOMOGRAPHY_CACHE_TTL_S`). Au-delà, la caméra est déclarée indisponible (elle a pu être déplacée). Toute nouvelle homographie est validée par **reprojection des coins détectés** : si le RMS > 2 cm (`HOMOGRAPHY_MAX_RMS_CM`), elle est rejetée et on retombe sur le cache récent avec un flag `stale=true`. Utile si un robot passe temporairement devant un coin ou si un faux positif fait déraper la calib.

### Précision réelle (ordre de grandeur)

| Métrique                        | Avant Phase 12 (homographie 2D) | **Après Phase 12 (solvePnP + auto-calib)** |
| ------------------------------- | ------------------------------- | ------------------------------------------ |
| Erreur position centre table    | ~±1 cm                          | **< 3 mm** (sur scène synthétique)         |
| Erreur position bords de table  | ~±1-2 cm                        | **< 1 cm** (ray-casting 3D exact)          |
| Jitter statique (frame à frame) | ±0.5 cm avec EMA α=0.4          | ±0.5 cm avec EMA α=0.4 (inchangé)          |
| Angle marqueur                  | ±1-2° (arctan 2D)               | **<1°** (extrait du rvec solvePnP)         |
| Framerate utile                 | ~15 fps (4K)                    | ~12-14 fps (4K, +solvePnP par marqueur)    |

Les trois sources de biais systématique sont supprimées :

1. **Distorsion radiale** — corrigée via `intrinsics.npz` obtenu par auto-calibration au démarrage (Phase 12.2).
2. **Homographie 2D plate** (noisette à 3 cm projetée comme si au sol) — remplacée par `solvePnP` par marqueur avec sa taille physique et sa hauteur (Phase 12.4 dans [markers.py](marker_detection/markers.py)).
3. **Constantes caméra codées en dur** (hauteur, position) — remplacées par la pose extrinsèque mesurée à chaque frame via `solvePnP` sur les 4 coins ArUco (Phase 12.3 dans [geometry.py](marker_detection/geometry.py)).

### Auto-calibration au démarrage (obligatoire)

La calibration intrinsèque est maintenant **obligatoire** et s'exécute automatiquement au premier lancement, **sans damier**. Les 4 coins ArUco de table (100×100 mm, positions 3D exactement connues) fournissent 16 correspondances 3D↔2D par frame ; sur ~10 frames accumulées, on estime `K` et les coefficients de distorsion, puis on persiste dans `intrinsics.npz`.

Flux au démarrage :

1. Si `intrinsics.npz` existe → chargé, pipeline démarre.
2. Sinon → phase warm-up : capture ~10 frames avec les 4 coins visibles, calibration (Zhang multi-vues si pose varie, sinon fallback « distorsion nulle + validation RMS < 1 px »), écriture de `intrinsics.npz`.
3. Si l'auto-calibration échoue (timeout 15 s ou RMS > 1 px) → **sortie avec erreur** demandant de lancer `tools/calibrate_camera.py` avec damier (fallback).

CLI :

```bash
python detect_markers.py                  # charge intrinsics.npz ou auto-calibre
python detect_markers.py --recalibrate    # force une nouvelle auto-calibration
python detect_markers.py --no-auto-calib  # echoue si intrinsics.npz absent
```

### Calibration avec damier (fallback)

La caméra smartphone (grand-angle typique) introduit une distorsion radiale non négligeable (±5 cm aux bords de table sans correction). Pour la compenser :

```bash
cd programs/vision
python tools/calibrate_camera.py          # imprime les contrôles clavier
```

Imprimer une mire échiquier **9×6 coins internes, case 30 mm** (par défaut), la présenter sous différentes orientations, capturer ≥10 poses avec **[espace]**, puis **[c]** pour lancer la calibration. Un `intrinsics.npz` est écrit à la racine de `vision/` et chargé automatiquement au prochain lancement de `detect_markers.py` (correction appliquée via `cv2.remap` en tête de pipeline).

Adapter les paramètres de la mire avec `--cols`, `--rows`, `--size-mm` si besoin.

---

## Tests

Une suite pytest couvre les briques algorithmiques (indépendantes de la caméra) :

```bash
cd programs/vision
nix develop --command python -m pytest tests/ -q
```

Modules testés :

- [tests/test_pose_smoother.py](tests/test_pose_smoother.py) — EMA position/angle, wrap ±180°, reset après timeout, labels dupliqués.
- [tests/test_pickup_zones.py](tests/test_pickup_zones.py) — appartenance ZR, fusion de slots, comptage garde-mangers.
- [tests/test_esp32_sender.py](tests/test_esp32_sender.py) — protocole texte `OPP/ZR/GM/STALE/CORNERS_OK/END` + clamping int16 / ±180° / ≤127.
- [tests/test_geometry.py](tests/test_geometry.py) — `compensate_height()` (fallback legacy).
- [tests/test_auto_calibration.py](tests/test_auto_calibration.py) — convergence auto-calib sur scène synthétique, rejet si frames insuffisantes ou coins manquants.
- [tests/test_solvepnp_position.py](tests/test_solvepnp_position.py) — précision sous-cm de `build_detected_list` avec pose caméra connue, au centre et aux bords de table.

## Mode replay (diagnostic hors caméra)

Pour rejouer une scène enregistrée (debug sans caméra, reproduction d'un bug de match) :

```bash
python detect_markers.py --replay-dir captures/2026-03-15/ --replay-fps 15
# --replay-once pour arrêter à la fin, sinon boucle
```

Les images `.jpg/.jpeg/.png/.bmp` du dossier sont lues dans l'ordre alphabétique au rythme `--replay-fps` et alimentent le pipeline comme une caméra live (ESP32, dashboard, MJPEG stream).

---

## Limitations connues

| Sujet                                     | État actuel                                                                                                                                          |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Calibration intrinsèque (distorsion)      | ✅ **Auto-calibrée au démarrage** depuis les 4 coins ArUco 100 mm, pas de damier ; fallback [`tools/calibrate_camera.py`](tools/calibrate_camera.py) |
| Pose extrinsèque caméra                   | ✅ Mesurée à chaque frame via `solvePnP` sur les 4 coins ([`estimate_camera_pose`](marker_detection/geometry.py))                                    |
| TTL sur le cache d'homographie            | ✅ 5 s ([`HOMOGRAPHY_CACHE_TTL_S`](marker_detection/config.py)) + flag `stale` propage                                                               |
| Validation d'homographie par reprojection | ✅ RMS < **1 cm** ([`HOMOGRAPHY_MAX_RMS_CM`](marker_detection/config.py))                                                                            |
| Validation intrinsèque auto-calib         | ✅ RMS < **1 px** ([`INTRINSIC_MAX_RMS_PX`](marker_detection/config.py))                                                                             |
| Lissage temporel des positions en cm      | ✅ EMA par label ([`PoseSmoother`](marker_detection/pose_smoother.py), α=0.4)                                                                        |
| Pose 3D par marqueur (`solvePnP`)         | ✅ **Actif** avec tailles 40/70/100 mm et hauteurs 3/45/0 cm ([`markers.py`](marker_detection/markers.py))                                           |
| Séquence noisettes par zone de ramassage  | ✅ [`PickupZoneTracker`](marker_detection/pickup_zones.py) + ligne `ZR id C1..C5`                                                                    |
| Comptage des noisettes par garde-manger   | ✅ `count_nuts_in_garde_mangers` + ligne `GM id count`                                                                                               |
| Position de l'adversaire avec angle       | ✅ Ligne `OPP x y θ×10` (+ legacy `Obstacle x y`)                                                                                                    |
| Relais dashboard → bridge en match        | ✅ `socket.ts` forward `Obstacle/OPP/ZR/GM/STALE/CORNERS_OK/END` à tous les robots                                                                   |
| Suite de tests automatisée                | ✅ `pytest tests/` (21 tests, cf. section _Tests_ ci-dessous)                                                                                        |
| Mode replay pour diagnostic hors caméra   | ✅ `detect_markers.py --replay-dir <dossier>` (images .jpg/.png)                                                                                     |

---

## Intégration firmware (Dave / PAMIs)

Côté robot, le parser `processCommand()` dans [`Commands.cpp`](../firmware/lib/EurobotCore/src/Commands.cpp) reconnaît `Obstacle`, `OPP`, `ZR`, `GM`, `STALE`, `CORNERS_OK`, `END` **avant** la cascade des commandes classiques (chemin rapide sans log verbose). Chaque ligne met à jour le singleton global [`g_visionState`](../firmware/lib/EurobotCore/src/VisionState.h).

API offerte à la stratégie :

```cpp
#include "VisionState.h"

if (g_visionState.hasOpponent()) {
    float d = g_visionState.opponentDistanceCm(state.robot_x, state.robot_y);
    if (d > 0 && d < 40.0f) { /* dégagement, re-routage, etc. */ }
}

// Lire la séquence d'une zone de récolte pour sauter le capteur RGB :
VisionNutColor top = g_visionState.zrTopColor(zoneId);
if (top == VISION_NUT_BLUE || top == VISION_NUT_YELLOW) {
    // couleur déjà connue → aller directement à l'aspiration sans scan RGB
}

// Garde-manger déjà vide ?
if (g_visionState.garde(gmId) == 0) { /* skip ce GM   */ }
```

Toutes les fonctions gèrent leur propre fraîcheur (`VISION_FRESHNESS_MS = 1500 ms` pour l'adversaire, ×10 pour les zones statiques). La stratégie doit toujours passer par les accesseurs — les champs publics sont destinés à un futur remplacement transparent par un `MSG_VISION_STATE` binaire si la latence devient critique.

## Commandes

### Avec Nix (recommandé)

Le `flake.nix` fournit Python 3.12 avec toutes les dépendances (OpenCV, NumPy, pyserial, python-socketio) + scrcpy + adb.

```bash
cd programs/vision
nix develop              # active le shell avec toutes les dépendances

# Définir l'équipe (optionnel, défaut = blue)
export TEAM_COLOR=blue      # ou yellow

# Définir l'URL du dashboard (optionnel, défaut = localhost:3001)
export DASHBOARD_URL=http://laptop-ip:3001

# Lancer la détection
python detect_markers.py

# Quitter : appuyer sur 'q' dans la fenêtre caméra
```

> **Depuis la racine du projet** : le `flake.nix` racine inclut aussi toutes les dépendances vision — `direnv allow` ou `nix develop` à la racine suffit.

### Sans Nix

```bash
pip install -r requirements.txt
export TEAM_COLOR=blue
python detect_markers.py
```

### Dépendances

| Package                   | Version | Usage                                           |
| ------------------------- | ------- | ----------------------------------------------- |
| `opencv-python`           | 4.13.0  | Capture, ArUco, homographie, affichage          |
| `numpy`                   | 2.4.2   | Calcul matriciel                                |
| `pyserial`                | 3.5     | Communication série vers ESP32                  |
| `python-socketio[client]` | 5.12.1  | Envoi des détections au dashboard via Socket.IO |

---

## Répertoires annexes

| Dossier             | Contenu                              |
| ------------------- | ------------------------------------ |
| `aruco/`            | PDFs imprimables des marqueurs ArUco |
| `marker_detection/` | Modules Python du pipeline           |
| `testing/`          | Scripts de test et validation        |
