# Audit de Performance — Pipeline Vision

**Date** : Avril 2026  
**Contexte** : ~3 secondes de latence observée entre le flux caméra réel et l'affichage annoté  
**Matériel** : OnePlus CPH2653 (Android 16) via scrcpy → /dev/video2, ASUS ZenBook UX481FA (i7-10510U, iGPU)  
**Résolution** : 3840×2160 @ 30 fps

---

## Résumé exécutif

Le pipeline traite chaque frame 4K de manière **séquentielle et mono-thread**, en pleine résolution, sans aucun skip de frames. La latence de ~3 s vient principalement de l'accumulation de frames non-lues dans le buffer V4L2 de `VideoCapture`. Chaque appel `cap.read()` renvoie la frame la plus ancienne du buffer, pas la plus récente.

**Gains estimés par correction** :

| Correction                               | Gain estimé                    | Complexité |
| ---------------------------------------- | ------------------------------ | ---------- |
| Flush du buffer capture (grab-only)      | **-2–3 s** de latence          | Trivial    |
| `DETECT_SCALE = 0.5` au lieu de `1.0`    | **×4 plus rapide** (détection) | 1 ligne    |
| Skip du fallback full-res quand coins OK | **-30–50 ms/frame**            | 5 lignes   |
| Désactiver QR (inutilisé en match)       | **-15–25 ms/frame**            | 1 flag     |
| Vue aérienne optionnelle                 | **-10–20 ms/frame**            | Flag       |
| Threading capture/traitement             | **-1 frame** de latence        | Modéré     |

---

## 1. Analyse détaillée des goulots d'étranglement

### 1.1 CRITIQUE — Buffer V4L2 non vidé (cause principale des 3 s)

**Fichier** : `detect_markers.py` L68  
**Code** :

```python
ret, frame = cap.read()
```

**Problème** : `VideoCapture.read()` lit la frame la plus ancienne du buffer interne V4L2/FFmpeg (typiquement 4–5 frames de profondeur). Si le traitement prend 100 ms et la caméra produit une frame toutes les 33 ms, le buffer se remplit. Après quelques secondes, la latence atteint `buffer_size × 33 ms + processing_time`, soit facilement 2–3 secondes.

**Fix** : vider le buffer avant de lire, ou utiliser `grab()` pour consommer les frames périmées :

```python
# Option A — grab/retrieve (simple, efficace)
cap.grab()  # consomme la frame la plus ancienne (quasi-instantané)
cap.grab()  # consomme la suivante
ret, frame = cap.read()  # lit la plus récente disponible

# Option B — thread dédié (optimal)
# Un thread fait cap.read() en continu et stocke la dernière frame.
# La boucle principale lit toujours self.latest_frame.
```

**Impact** : cette seule correction devrait réduire la latence de ~3 s à ~100–200 ms.

### 1.2 MAJEUR — `DETECT_SCALE = 1.0` (détection en pleine résolution 4K)

**Fichier** : `config.py` L49  
**Code** :

```python
DETECT_SCALE = 1
```

**Problème** : le resize dans `detect_all()` est un no-op. La détection ArUco s'exécute sur 3840×2160 pixels = 8.3 Mpx. Le coût est quasi-linéaire avec le nombre de pixels.

À `DETECT_SCALE = 0.5` (1920×1080), on traite 4× moins de pixels. Les marqueurs ArUco de 70 mm (robots) font ~80 px en 4K ; à 0.5× ils font ~40 px, largement suffisant pour `DICT_4X4_50`. Les marqueurs de 40 mm (noisettes) font ~45 px en 4K → ~22 px à 0.5×, encore détectables.

**Fix** : changer `DETECT_SCALE = 0.5` (ou `0.25` pour du 960×540, suffisant pour les gros marqueurs).

**Benchmark estimé** :

| Scale | Résolution | Pixels | Temps détection ArUco |
| ----- | ---------- | ------ | --------------------- |
| 1.0   | 3840×2160  | 8.3M   | ~60–100 ms            |
| 0.5   | 1920×1080  | 2.1M   | ~15–25 ms             |
| 0.25  | 960×540    | 0.5M   | ~5–10 ms              |

### 1.3 MAJEUR — Fallback full-res systématique

**Fichier** : `detection.py` L76–86  
**Code** :

```python
found_corners = {mid for mid in aruco_ids if mid in config.CORNER_IDS}
if found_corners != config.CORNER_IDS:
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    enhanced_full = clahe.apply(frame_gray)
    raw_corners_full, raw_ids_full, _ = detector.detectMarkers(enhanced_full)
```

**Problème** : si un seul des 4 coins de table est manquant (occlusion partielle, reflet), le pipeline relance une détection ArUco sur la **totalité** de l'image en pleine résolution. Avec `DETECT_SCALE = 1.0`, c'est un deuxième scan 4K. Même avec un scale de 0.5, le fallback en full-res double le temps.

**Coût** : +60–100 ms à chaque frame où un coin manque (fréquent en pratique avec les reflets/occultations).

**Fix** :

- Ne déclencher le fallback que si les coins n'ont pas été vus depuis N frames (pas à chaque frame manquante)
- Faire le fallback sur un ROI (zone attendue du coin manquant) plutôt que l'image entière
- Utiliser le tracker : les 4 coins sont fixes, une fois validés, le fallback est inutile

### 1.4 MODÉRÉ — GaussianBlur 4K redondant

**Fichier** : `detection.py` L63, L78  
**Code** :

```python
small = cv2.GaussianBlur(small, (5, 5), 0)   # ligne 63 — sur small
# ...
frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)  # ligne 78 — sur full 4K
```

**Problème** : un blur 5×5 sur 3840×2160 prend ~8–12 ms. Le CLAHE qui suit fait déjà un filtrage adaptatif. Le blur est redondant avec CLAHE pour la plupart des cas d'utilisation.

**Fix** : supprimer le blur de la passe full-res ou le réduire à 3×3.

### 1.5 MODÉRÉ — `validate_aruco()` crée un masque plein-cadre par candidat

**Fichier** : `detection.py` L34–38  
**Code** :

```python
mask = np.zeros(gray.shape, dtype=np.uint8)  # alloue 8.3 MB à chaque appel en 4K
cv2.fillPoly(mask, [pts], 255)
pixels = gray[mask > 0]
```

**Problème** : pour chaque marqueur candidat (~5–20 par frame), on alloue un masque de la taille de l'image complète (3840×2160 = 8.3 MB), on dessine un polygone, puis on indexe. Avec 10 candidats : 83 MB alloués et libérés par frame.

**Fix** : utiliser le bounding rect du marqueur et ne masquer que le ROI :

```python
x, y, w, h = cv2.boundingRect(pts)
roi = gray[y:y+h, x:x+w]
pts_local = pts - [x, y]
mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(mask, [pts_local], 255)
pixels = roi[mask > 0]
```

**Gain** : ~5–15 ms/frame (suppression de ~80 MB d'allocations).

### 1.6 MODÉRÉ — `warpPerspective` 4K pour la vue aérienne

**Fichier** : `visualization.py` L90–93  
**Code** :

```python
aerial = cv2.warpPerspective(frame, h_aerial, (config.AERIAL_W, config.AERIAL_H))
```

**Problème** : la transformation de perspective d'une image 4K vers 1200×800 prend ~10–20 ms. C'est fait une frame sur deux, donc ~5–10 ms en moyenne.

**Fix** :

- Resize l'image avant le warp (on n'a besoin que de 1200×800 de sortie)
- Rendre la vue aérienne optionnelle (flag `--no-aerial`)
- Ne la calculer que toutes les 10 frames (elle ne change pas significativement)

### 1.7 MODÉRÉ — Overlays dessinés sur l'image 4K

**Fichier** : `visualization.py` (multiple)  
**Problème** : `draw_grid()`, `draw_table_outline()`, `draw_status()`, `draw_detection()` dessinent directement sur le frame 4K. Les `cv2.line()`, `cv2.polylines()`, `cv2.putText()` sur une image 4K sont plus lents que sur une image 1080p.

`draw_grid()` dessine ~20 lignes avec des projections de perspective pour chaque ligne.

**Fix** : resize le frame avant le rendu overlay, ou rendre les overlays sur un calque plus petit puis superposer.

### 1.8 MODÉRÉ — `imshow` sur des images 4K

**Fichier** : `detect_markers.py` L126  
**Code** :

```python
cv2.imshow(config.WINDOW_CAMERA, frame)
```

**Problème** : afficher une image 3840×2160 via `imshow` (GTK backend, sans accélération GPU) prend ~15–30 ms. La fenêtre est redimensionnée à `WINDOW_CAMERA_SIZE = (3840, 2160)` — c'est la taille native, donc pas de downscale implicite par GTK.

**Fix** : resize le frame à 1920×1080 (ou 1280×720) avant `imshow`. L'écran du ZenBook fait probablement 1920×1080 ; afficher du 4K natif est inutile.

### 1.9 MINEUR — Détection QR probablement inutile

**Fichier** : `detection.py` L88–95  
**Code** :

```python
retval, decoded, points, _ = qr_detector.detectAndDecodeMulti(enhanced_small)
```

**Problème** : `detectAndDecodeMulti` est significativement plus coûteux qu'ArUco seul (~10–25 ms même sur l'image réduite). Les QR codes ne semblent pas utilisés en match Eurobot 2026 (aucune caisse ou zone n'utilise de QR).

**Fix** : désactiver par défaut, activer via flag `--enable-qr`.

### 1.10 MINEUR — `imshow` pour la vue aérienne

`WINDOW_AERIAL_SIZE = (800, 600)` est raisonnable, mais l'appel `imshow` + `waitKey` ajoutent ~5 ms combinés.

---

## 2. Architecture — Problèmes structurels

### 2.1 Pipeline mono-thread sans pipelining

```
Capture → Grayscale → Detect → Track → Transform → Visualize → imshow → waitKey
   │                                                                         │
   └─────────────────── 100–200 ms total ────────────────────────────────────┘
```

Tout est séquentiel. Pendant le traitement (~100 ms), la caméra continue de produire des frames qui s'empilent dans le buffer.

**Architecture cible** :

```
Thread 1 (capture)  : cap.read() en boucle → latest_frame (atomique)
Thread 2 (pipeline) : lit latest_frame → detect → track → send
Thread 3 (display)  : affiche le résultat (optionnel, faible priorité)
```

### 2.2 Pas de mesure de performance

Aucun timing n'est mesuré dans le code. Il est impossible de savoir quelle étape prend combien de temps sans ajouter des `time.perf_counter()` manuellement.

**Fix** : ajouter un mode `--profile` qui affiche les timings par étape et le FPS résultant.

### 2.3 Homographie recalculée à chaque frame

**Fichier** : `detect_markers.py` L86  
`build_transforms()` est appelé à chaque frame. Les 4 coins de table sont **fixes** (collés sur la table). Une fois l'homographie calculée et validée sur quelques frames, elle ne devrait plus changer.

**Fix** : cacher l'homographie après N frames consécutives avec les 4 coins détectés. Ne recalculer que si un coin bouge significativement (> 5 px).

### 2.4 Socket.IO synchrone dans la boucle principale

**Fichier** : `dashboard_sender.py` L96  
`self._sio.emit()` est appelé dans la boucle principale. Si le serveur dashboard est lent ou déconnecté, l'emit peut bloquer ou ajouter de la latence.

**Fix** : émettre dans un thread séparé ou utiliser le client async (`socketio.AsyncClient`).

### 2.5 Communication série synchrone

**Fichier** : `esp32_sender.py`  
`serial.write()` est synchrone. Avec 10 marqueurs à 115200 baud, l'envoi prend ~1–2 ms (négligeable), mais un timeout ou une erreur série peut bloquer la boucle.

---

## 3. Configuration sous-optimale

### 3.1 `WINDOW_CAMERA_SIZE = (3840, 2160)`

La fenêtre d'affichage est configurée à la taille native 4K. Sur un écran 1080p (ZenBook), c'est inutile et force un downscale par le window manager.

### 3.2 `AERIAL_W = 1200, AERIAL_H = 800`

La vue aérienne est inutilement grande. 600×400 suffirait pour le debug.

### 3.3 Pas de `CAP_PROP_BUFFERSIZE`

Le nombre de frames en buffer n'est pas configuré :

```python
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise la latence
```

Ce flag n'est pas toujours supporté par V4L2, mais quand il l'est, il réduit drastiquement la latence.

---

## 4. Plan d'optimisation recommandé

### Phase 1 — Quick wins (< 30 min, gain estimé : latence de 3 s → ~150 ms)

| #      | Action                                                                     | Fichier             | Impact             |
| ------ | -------------------------------------------------------------------------- | ------------------- | ------------------ |
| **P1** | Flush le buffer V4L2 avant `read()` (grab loop ou `CAP_PROP_BUFFERSIZE=1`) | `detect_markers.py` | **-2–3 s latence** |
| **P2** | `DETECT_SCALE = 0.5`                                                       | `config.py`         | **×4 détection**   |
| **P3** | Resize frame à 1080p avant overlays et `imshow`                            | `detect_markers.py` | **-20–40 ms**      |
| **P4** | Masque ROI au lieu de plein-cadre dans `validate_aruco`                    | `detection.py`      | **-10 ms**         |

### Phase 2 — Optimisations ciblées (< 2h)

| #       | Action                                           | Fichier                           | Impact                                |
| ------- | ------------------------------------------------ | --------------------------------- | ------------------------------------- |
| **P5**  | Cacher l'homographie (coins fixes)               | `detect_markers.py`               | -5 ms + supprime le fallback full-res |
| **P6**  | Désactiver QR par défaut                         | `detection.py`, `config.py`       | -15–25 ms                             |
| **P7**  | Supprimer le blur full-res dans le fallback      | `detection.py`                    | -8–12 ms                              |
| **P8**  | Vue aérienne toutes les 10 frames ou optionnelle | `visualization.py`                | -10 ms avg                            |
| **P9**  | Ajouter `--profile` pour mesurer les timings     | `detect_markers.py`               | Debug                                 |
| **P10** | Ajouter `--headless` pour tourner sans GUI       | `detect_markers.py`, `runtime.py` | NixOS sans GTK                        |

### Phase 3 — Architecture (< 1 jour)

| #       | Action                                          | Impact                                   |
| ------- | ----------------------------------------------- | ---------------------------------------- |
| **P11** | Thread de capture séparé (latest-frame pattern) | -1 frame de latence, throughput constant |
| **P12** | Thread d'affichage séparé                       | Le display ne bloque plus le pipeline    |
| **P13** | Socket.IO async ou thread séparé                | Supprime le couplage avec le dashboard   |

---

## 5. Estimation des performances après optimisation

| Métrique             | Avant           | Après Phase 1 | Après Phase 2 | Après Phase 3 |
| -------------------- | --------------- | ------------- | ------------- | ------------- |
| Latence bout-en-bout | ~3 s            | ~150 ms       | ~100 ms       | ~70 ms        |
| FPS traitement       | ~5–8            | ~15–20        | ~25–30        | ~30           |
| CPU (i7-10510U)      | ~80–100% 1 core | ~40% 1 core   | ~25% 1 core   | ~40% 2 cores  |

---

## 6. Résumé des priorités

```
┌─────────────────────────────────────────────────────────┐
│ CRITIQUE  (cause 90% de la latence)                     │
│                                                         │
│  P1  Buffer V4L2 non vidé — frames périmées empilées    │
│  P2  DETECT_SCALE = 1.0 — détection 4K inutile          │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ MAJEUR  (gain 30–60 ms/frame)                           │
│                                                         │
│  P3  imshow / overlays en 4K                            │
│  P4  validate_aruco masque plein-cadre                  │
│  P5  Homographie recalculée inutilement                 │
│  P6  Détection QR inutile en match                      │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ MODÉRÉ  (architecture long-terme)                       │
│                                                         │
│  P11 Thread capture séparé                              │
│  P10 Mode headless (NixOS sans GTK)                     │
│  P9  Profiling intégré                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
