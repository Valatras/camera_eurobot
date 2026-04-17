# Audit du Système de Vision

**Date** : Juin 2025  
**Scope** : `programs/vision/` — pipeline de détection ArUco pour Eurobot 2025  
**Matériel cible** : OnePlus 13 via Iriun Webcam USB → laptop (GPU disponible) en zone déportée  
**Objectifs** : Position, couleur et orientation de toutes les noisettes + position des ennemis et de nos robots via ArUco

---

## 1. Inventaire de l'existant

### 1.1 Stack technique

| Composant          | Version / Détail          |
| ------------------ | ------------------------- |
| Python             | 3.12                      |
| OpenCV             | 4.13.0                    |
| NumPy              | 2.4.2                     |
| pyserial           | 3.5                       |
| Dictionnaire ArUco | `DICT_4X4_50`             |
| Résolution cible   | 3840×2160 @ 30 fps (V4L2) |

### 1.2 Architecture du pipeline

```
capture (V4L2) → grayscale → detect_all (ArUco multi-pass + QR)
    → tracking (buffer=3, min_hits=2)
    → separate_markers (coins table vs objets)
    → build_transforms (4 coins → homographie)
    → build_detected_list (label, grid_x, grid_y)
    → esp32_sender (serial ASCII) + dashboard_sender (HTTP JSON)
    → visualization (overlays + vue aérienne)
```

### 1.3 Modules existants

| Fichier               | Rôle                                             | Etat                       |
| --------------------- | ------------------------------------------------ | -------------------------- |
| `detect_markers.py`   | Boucle principale                                | Fonctionnel                |
| `config.py`           | Constantes (résolution, IDs, grille)             | Fonctionnel                |
| `detection.py`        | Détection multi-pass + validation                | Fonctionnel                |
| `geometry.py`         | Homographie 4 coins, perspective, projection     | Fonctionnel                |
| `markers.py`          | Classification IDs, séparation, liste détections | Fonctionnel mais incomplet |
| `tracking.py`         | Lissage temporel (moyenne glissante)             | Fonctionnel mais basique   |
| `runtime.py`          | Init caméra, détecteur ArUco, CLAHE              | Fonctionnel                |
| `esp32_sender.py`     | Communication série auto-detect                  | Fonctionnel                |
| `dashboard_sender.py` | POST HTTP JSON                                   | Fonctionnel                |
| `visualization.py`    | Overlays + vue aérienne                          | Fonctionnel                |
| `testing/`            | Probe caméra, tests résolution Iriun             | Utilitaires                |

### 1.4 Ce qui fonctionne

- **Détection ArUco** : multi-pass (résolution réduite + fallback pleine résolution pour les coins), validation (aire, aspect ratio, variance pixel), CLAHE pour conditions d'éclairage variables.
- **Homographie** : 4 marqueurs de coin (IDs 20–23, inset 500 mm) → extrapolation des bords réels de la table → matrices de perspective image↔grille et image→aérien.
- **Grille** : 30×20 cellules = 100 mm × 100 mm par cellule.
- **Tracking temporel** : buffer de 3 frames, 2 hits minimum, moyenne des coins.
- **Communication** : série ASCII `TYPE,X,Y\n` + `END\n` vers ESP32, HTTP JSON vers dashboard.
- **Intégration firmware** : `MSG_OPPONENT_POS` (0x04) via ESP-NOW → `onOpponentPosition()` → mise à jour du costmap.

---

## 2. Analyse des écarts vs objectifs

### 2.1 Identification des noisettes — IDs ArUco incorrects

**Règlement** (§D.4) :

- Caisses de noisettes : ArUco **36** (bleues), **47** (jaunes), **41** (vides/noires)
- Ces IDs sont dans la plage 11–50 réservée à l'aire de jeu

**Code actuel** (`markers.py`) :

```python
if 11 <= marker_id <= 50 and marker_id not in config.CORNER_IDS:
    return f"AREA{marker_id}"
```

**Problème** : les IDs 36, 47 et 41 sont classés comme `AREA36`, `AREA47`, `AREA41` — indistinguables des marqueurs de zone. Le firmware et le dashboard ne peuvent pas distinguer une caisse bleue d'une caisse jaune ou d'une caisse vide.

**Correction nécessaire** : ajouter des constantes `NUT_BLUE_ID = 36`, `NUT_YELLOW_ID = 47`, `NUT_EMPTY_ID = 41` et les classifier en `NUT_BLUE`, `NUT_YELLOW`, `NUT_EMPTY`.

| Sévérité     | Impact                                                                     |
| ------------ | -------------------------------------------------------------------------- |
| **CRITIQUE** | Sans cette distinction, aucune stratégie de tri des caisses n'est possible |

> **Note couleur** : la couleur de chaque caisse est directement déduite de l'ID ArUco (36 → bleue, 47 → jaune, 41 → vide/noire). Aucune analyse colorimétrique (HSV) n'est nécessaire — la correction de la classification des IDs (§2.1) résout simultanément l'identification de couleur. C'est l'approche la plus fiable et la plus rapide (zéro latence ajoutée).

### 2.3 Pas d'extraction d'orientation

**Objectif** : déterminer l'orientation (angle) de chaque caisse.

**État actuel** : seul le centre du marqueur est projeté en coordonnées grille. Les 4 coins du marqueur sont disponibles dans la détection mais l'angle n'est jamais calculé.

**Ce qui manque** :

- Calcul de l'angle à partir des coins du marqueur (vecteur entre les 2 premiers coins → `atan2`)
- Ou estimation de pose complète via `cv2.solvePnP` avec la taille physique connue (40 mm)
- Retour d'un champ `angle_deg` dans les données

| Sévérité   | Impact                                                            |
| ---------- | ----------------------------------------------------------------- |
| **MAJEUR** | Nécessaire pour planifier l'approche et la préhension des caisses |

### 2.4 Pas de localisation propre des robots alliés

**Objectif** : connaître la position de nos propres robots via ArUco.

**État actuel** : `classify_marker_id()` distingue bien `BR1-5` (robots bleus) et `YR1-5` (robots jaunes), mais :

- Les positions sont envoyées uniquement en coordonnées grille entières (résolution 100 mm)
- Le firmware ne consomme que `MSG_OPPONENT_POS` — aucun message `MSG_OWN_POS` n'existe
- Aucune fusion n'est faite entre la position vue par la caméra et l'odométrie embarquée

**Ce qui manque** :

- Envoi de la position des robots alliés au firmware (nouveau type de message)
- Position en mm (pas en cellules de grille) pour être utile à la navigation
- Orientation du robot (calculable depuis les 4 coins du marqueur de 70 mm)

| Sévérité   | Impact                                                                |
| ---------- | --------------------------------------------------------------------- |
| **MAJEUR** | La localisation externe permettrait de corriger la dérive odométrique |

### 2.5 Résolution de sortie insuffisante

**État actuel** : la grille est 30×20 cellules → 100 mm par cellule. Les coordonnées envoyées sont des entiers (`int(round(pos))`).

**Problème** : une résolution de 100 mm est inadmissible pour :

- La préhension de caisses de 150×50 mm (il faut ~10 mm de précision)
- La localisation de robots (le costmap reçoit des cellules de 100 mm)
- La détection de collision (marge insuffisante)

**Correction** : envoyer les positions en millimètres (float) plutôt qu'en cellules de grille.

| Sévérité   | Impact                                                                      |
| ---------- | --------------------------------------------------------------------------- |
| **MAJEUR** | Perte de précision d'un facteur 10 par rapport au potentiel de la caméra 4K |

### 2.6 Pas de calibration caméra

**État actuel** : aucune calibration intrinsèque (matrice K, coefficients de distorsion). L'homographie compense partiellement la perspective, mais pas la distorsion radiale/tangentielle.

**Ce qui manque** :

- Procédure de calibration (damier/ChArUco) → fichier JSON/YAML de paramètres
- `cv2.undistort()` ou `cv2.initUndistortRectifyMap()` + `cv2.remap()` en début de pipeline
- Paramètres spécifiques à l'Iriun Webcam (qui ajoute sa propre couche de compression/déformation)

**Impact** : erreurs de position croissantes en bord de champ (potentiellement 20-50 mm à 4K avec un champ large depuis 1.6 m de haut). L'homographie seule ne corrige pas la distorsion de la lentille.

| Sévérité   | Impact                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------ |
| **MODÉRÉ** | Introduit des erreurs systématiques en bord de table, partiellement compensées par l'homographie |

### 2.7 Tracking trop simpliste

**État actuel** : `Tracker` fait une moyenne glissante sur 3 frames avec 2 hits minimum. Pas d'estimation de vitesse ni de prédiction.

**Limitations** :

- Un robot à 500 mm/s parcourt 33 mm entre 2 frames (à 15 fps effectifs). La moyenne retarde la position.
- Pas de gestion d'occlusion (un marqueur masqué 2 frames est perdu immédiatement)
- Pas de réassociation quand un marqueur réapparaît après occlusion
- Pas de filtrage de Kalman pour prédire la position en cas de détection manquée

**Upgrade recommandé** : Kalman filter par marqueur (état = [x, y, vx, vy]) avec prédiction à chaque frame et mise à jour quand la détection est disponible.

| Sévérité   | Impact                                                                                   |
| ---------- | ---------------------------------------------------------------------------------------- |
| **MODÉRÉ** | Positions décalées pour les objets en mouvement, perte de suivi en cas d'occlusion brève |

### 2.8 Protocole série limité

**État actuel** : `TYPE,X,Y\n` — texte ASCII, pas d'orientation, pas de couleur, pas de confiance, pas de timestamp.

**Ce qui manque pour les objectifs** :

- Champ orientation (angle en degrés)
- Champ couleur (pour les caisses)
- Champ confiance (nombre de frames de tracking)
- Format binaire optionnel pour réduire la latence (un paquet de 10 marqueurs = ~150 octets en ASCII vs ~60 en binaire)
- Côté firmware : parser étendu + nouveaux types de messages CAN-like pour les caisses et robots alliés

| Sévérité   | Impact                                                 |
| ---------- | ------------------------------------------------------ |
| **MODÉRÉ** | Bloque l'envoi des données d'orientation et de couleur |

---

## 3. Conformité réglementaire

### 3.1 Zone déportée (§G.4)

| Contrainte                                                           | Conformité        | Note                                                  |
| -------------------------------------------------------------------- | ----------------- | ----------------------------------------------------- |
| Plateforme sur l'axe de symétrie, bord arrière                       | ⚠️ À vérifier     | Le laptop + téléphone doivent tenir sur la plateforme |
| Volume : 450×320 mm en surface, max 1.6 m au-dessus de la plateforme | ⚠️ À vérifier     | Hauteur du trépied/support téléphone                  |
| Masse < 5 kg                                                         | ⚠️ À vérifier     | Laptop + support + câbles                             |
| Fixation : tige filetée M8 + écrou papillon dans rainure 10 mm       | ❌ Non implémenté | Nécessaire pour l'homologation                        |
| Pas de dépassement côté adverse                                      | ✅ Implicite      | Un seul côté de plateforme                            |
| Vibrations possibles                                                 | ⚠️ À considérer   | Fixation anti-vibration du téléphone                  |

### 3.2 Identification robot (§G.6)

| Contrainte                                                                 | Conformité                                                       |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Marqueur 10 cm, ArUco 4x4 70 mm, marge blanche 10 mm, contour couleur 5 mm | ✅ Fourni par l'organisation                                     |
| Bleus = IDs 1–5, Jaunes = IDs 6–10                                         | ✅ Correctement classifié dans `markers.py`                      |
| Support de balise requis pour marqueur                                     | ⚠️ Mécanique, hors scope vision                                  |
| ArUco 4x4 IDs 0–50 interdits sur les robots                                | ✅ Le code n'utilise pas d'ArUcos personnalisés dans cette plage |

### 3.3 Éléments de jeu (§D.4)

| Élément       | ID ArUco | Taille tag | Reconnu par le code | Classification                                   |
| ------------- | -------- | ---------- | ------------------- | ------------------------------------------------ |
| Caisse bleue  | 36       | 40 mm      | ✅ Détecté          | ❌ Classé comme `AREA36` au lieu de `NUT_BLUE`   |
| Caisse jaune  | 47       | 40 mm      | ✅ Détecté          | ❌ Classé comme `AREA47` au lieu de `NUT_YELLOW` |
| Caisse vide   | 41       | 40 mm      | ✅ Détecté          | ❌ Classé comme `AREA41` au lieu de `NUT_EMPTY`  |
| Coin table TL | 23       | —          | ✅                  | ✅                                               |
| Coin table TR | 22       | —          | ✅                  | ✅                                               |
| Coin table BR | 20       | —          | ✅                  | ✅                                               |
| Coin table BL | 21       | —          | ✅                  | ✅                                               |

---

## 4. Risques spécifiques Iriun Webcam

| Risque                | Description                                                                                                                                               | Mitigation                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Latence**           | Iriun ajoute une couche d'encodage/décodage vidéo par-dessus USB. Latence mesurée typiquement 50–150 ms en plus du pipeline de détection                  | Utiliser le mode USB direct (pas WiFi), forcer MJPEG plutôt que H.264 dans les settings Iriun |
| **Résolution réelle** | 4K annoncé mais la résolution effective dépend de la compression Iriun. Le `testing/resolution.py` existing probe déjà les résolutions — bon réflexe      | Valider le signal réel avec un test de mire/damier                                            |
| **V4L2 backend**      | Iriun crée un device `/dev/videoN` via v4l2loopback. Compatible mais parfois instable                                                                     | Fallback sur index 0–4 déjà implémenté dans `runtime.py`                                      |
| **Stabilité USB**     | Déconnexions possibles, surtout avec vibrations en zone déportée                                                                                          | Ajouter un mécanisme de reconnexion automatique de la caméra                                  |
| **Champ de vision**   | Depuis 1.6 m de haut, un capteur 4K couvre une table de 3×2 m avec un FOV d'environ 80–90°. Un tag de 40 mm fait ~80 pixels au centre, ~50 pixels en bord | Suffisant pour la détection ArUco, mais marginal                                              |

---

## 5. Évaluation GPU

Le GPU du laptop **n'est pas nécessaire** pour le pipeline actuel (ArUco pur en CPU). Il deviendrait pertinent si :

| Scénario                                                 | GPU utile ?                                                             |
| -------------------------------------------------------- | ----------------------------------------------------------------------- |
| Pipeline actuel (ArUco + homographie)                    | Non — OpenCV ArUco est CPU-only, 4K@15fps atteignable sur un i5/Ryzen 5 |
| Ajout de segmentation couleur (HSV)                      | Non — opération pixel triviale, < 5 ms sur CPU en 4K                    |
| Détection de caisses sans ArUco (deep learning YOLO/SSD) | **Oui** — nécessaire pour inference temps réel                          |
| Pose estimation (solvePnP) par marqueur                  | Non — quelques µs par appel, négligeable                                |
| Tracking multi-objet avancé (DeepSORT)                   | **Oui** — si le réidentification CNN est utilisée                       |

**Recommandation** : garder le pipeline CPU-only pour Eurobot. Le GPU n'est pertinent que si les ArUcos sur les caisses sont illisibles en pratique (occultation, vinyle abîmé) et qu'il faut basculer en détection par réseau de neurones.

---

## 6. Intégration Dashboard — État & Plan

### 6.1 État actuel — Problèmes critiques

L'exploration du code dashboard (`apps/server/`, `apps/web/`) révèle **4 ruptures** dans la chaîne de données :

| #      | Problème                                         | Détail                                                                                                                                                                                                                                             |
| ------ | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **D1** | **Endpoint `/api/marker-detections` inexistant** | `DashboardSender` POST vers `http://localhost:3001/api/marker-detections` — cette route n'existe pas dans `apps/server/src/index.ts`. Résultat : **404 systématique**.                                                                             |
| **D2** | **`CameraManager` lance le mauvais script**      | `camera.ts` résout `../../eye_of_Sauron/detect_squares.py` (chemin obsolète), pas `programs/vision/detect_markers.py`.                                                                                                                             |
| **D3** | **Détections caméra jamais relayées aux robots** | Les détections camera mettent à jour `store.opponent` pour l'UI, mais le serial `Obstacle X Y` n'est envoyé que pour les obstacles manuels (`addObstacle`). `serializeOpponentPos` est importé mais **jamais utilisé** pour les détections caméra. |
| **D4** | **Noisettes non affichées sur la carte**         | `CameraFeed.tsx` affiche une liste texte. `MapCanvas` ne rend que les obstacles manuels (`nut_crate`, `opponent_robot`), pas les détections caméra automatiques.                                                                                   |

### 6.2 Architecture cible

```
Vision pipeline (Python)
    │
    ├── WebSocket (Socket.IO) ──→ Dashboard server ──→ Dashboard UI (MapCanvas)
    │                                   │
    │                                   └──→ Serial → Bridge ESP32 → ESP-NOW → Robots
    │
    └── Serial USB (backup) ──→ Bridge ESP32 (direct, sans dashboard)
```

**Choix d'architecture** : le dashboard server devient le hub central. Le pipeline vision envoie ses détections via Socket.IO (pas HTTP POST), le server les redistribue au frontend ET aux robots.

### 6.3 Plan d'intégration

#### Phase D-1 : Connexion vision → dashboard (Socket.IO)

| Tâche                                                    | Fichiers                                      | Détail                                                                                            |
| -------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Remplacer `DashboardSender` HTTP par un client Socket.IO | `vision/marker_detection/dashboard_sender.py` | Émettre un événement `visionDetections` avec `{markers: [{label, x_mm, y_mm, angle_deg}], ts_ms}` |
| Récepteur côté server                                    | `dashboard/apps/server/src/socket.ts`         | Écouter `visionDetections`, mettre à jour `store`, broadcaster vers clients web                   |
| Supprimer `CameraManager` (subprocess)                   | `dashboard/apps/server/src/camera.ts`         | Le pipeline tourne indépendamment, plus besoin de le spawner                                      |

#### Phase D-2 : Affichage noisettes sur MapCanvas

| Tâche                                     | Fichiers                                          | Détail                                                                          |
| ----------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------- | ---------- |
| Nouveau store `cameraDetections` typé     | `dashboard/packages/shared/`                      | Types: `NutDetection {id, x_mm, y_mm, angle_deg, color: 'blue'                  | 'yellow' | 'empty'}`, `RobotDetection {id, x_mm, y_mm, angle_deg, team: 'blue' | 'yellow'}` |
| Layer noisettes dans `MapCanvas`          | `dashboard/apps/web/src/components/MapCanvas.tsx` | Rectangles 15×5 cm, couleur selon type, flèche d'orientation, opacité selon âge |
| Layer adversaires caméra dans `MapCanvas` | `MapCanvas.tsx`                                   | Remplacer le cercle rouge unique par N adversaires dynamiques depuis la caméra  |
| Indicateur de fraîcheur                   | `MapCanvas.tsx`                                   | Badge "CAM" vert/rouge selon heartbeat des détections (seuil 2s)                |

#### Phase D-3 : Relay détections → firmware

| Tâche                                       | Fichiers                              | Détail                                                                                                           |
| ------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Relayer positions adversaires en temps réel | `dashboard/apps/server/src/socket.ts` | Sur réception de `visionDetections`, extraire les robots adverses, envoyer `@all Obstacle X Y` via serial bridge |
| Relayer positions noisettes (optionnel)     | `socket.ts`, firmware `Commands.cpp`  | Nouveau message `MSG_NUT_POS` si la stratégie a besoin de connaître les caisses restantes                        |
| Protocole étendu noisettes                  | `CommDefs.h`                          | `NutPosMsg {id: u8, x_cm: u16, y_cm: u16, angle_deg: i16, color: u8}` — 8 bytes                                  |

### 6.4 Protocole de données vision → dashboard

```json
{
  "nuts": [
    { "id": 36, "x_mm": 1250, "y_mm": 830, "angle_deg": 45.2, "color": "blue" },
    {
      "id": 47,
      "x_mm": 750,
      "y_mm": 1200,
      "angle_deg": -12.0,
      "color": "yellow"
    }
  ],
  "robots": [
    { "id": 1, "x_mm": 450, "y_mm": 600, "angle_deg": 90.0, "team": "blue" },
    {
      "id": 7,
      "x_mm": 2100,
      "y_mm": 1400,
      "angle_deg": 270.0,
      "team": "yellow"
    }
  ],
  "corners_ok": true,
  "fps": 14.8,
  "ts_ms": 1712678400000
}
```

Catégorisation automatique par l'ID ArUco :

- `id ∈ {36}` → `color: "blue"` (déduit de l'ID, pas de HSV)
- `id ∈ {47}` → `color: "yellow"`
- `id ∈ {41}` → `color: "empty"`
- `id ∈ [1,5]` → `team: "blue"`, `id ∈ [6,10]` → `team: "yellow"`

---

## 7. Aide à la navigation — État & Plan

### 7.1 État actuel de la chaîne navigation

| Composant                   | Détail                                                                                                                                 |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Odométrie**               | Encodeurs + IMU (gyro) fusionnés via `AngleFusion` (filtre complémentaire). Midpoint integration. Stocké dans `state.robot_x/y/theta`. |
| **Costmap**                 | Grille `60×40`, cellule = 5 cm. Obstacles statiques (bordures, grenier) + inflation runtime.                                           |
| **Adversaire sur costmap**  | `updateOpponent(x_cm, y_cm)` → disque de 20 cm + inflation. Clear automatique après `CAMERA_TIMEOUT_MS` (2s).                          |
| **Capteurs US sur costmap** | Jusqu'à 4 obstacles simultanés, LRU, disque 20 cm, clearing par âge.                                                                   |
| **DWA Planner**             | 8×11 échantillons (v,ω), sim 0.5s, poids heading=3/clearance=2/velocity=1. Check `isPositionFree()` sur disque `getNavRobotRadius()`.  |
| **Reactive Avoid**          | 3 phases (slow→stop→stuck), hystérésis 5 cm, commitment 500 ms, grace 800 ms.                                                          |
| **Correction externe**      | `SetPose x y θ` — overwrite brut, utilisé uniquement au départ. **Aucune correction continue.**                                        |

### 7.2 Problèmes identifiés

| #      | Problème                                     | Impact                                                                                                                                                                                                                        |
| ------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **N1** | **Pas de correction de position continue**   | L'odométrie dérive inévitablement (glissement roues, chocs). Après 90s de match, l'erreur peut atteindre 5–15 cm. La caméra voit les robots à ~10 mm près mais cette info n'est pas exploitée.                                |
| **N2** | **Un seul adversaire supporté**              | `updateOpponent()` efface le blob précédent avant d'en poser un nouveau. Si 2 adversaires sont sur la table (robot + PAMI), seul le dernier reçu est sur le costmap.                                                          |
| **N3** | **Pas de position noisettes sur le costmap** | Les caisses (150×50 mm) sont des obstacles potentiels mais ne sont pas injectées dans le costmap. Le robot peut planifier un chemin à travers une caisse.                                                                     |
| **N4** | **Latence vision → costmap non mesurée**     | Chaîne : caméra (50–150 ms Iriun) + détection (~30 ms) + tracking (2 frames × 33 ms) + serial (~5 ms) + ESP-NOW (~10 ms). Total estimé : **200–350 ms**. À cette latence, un adversaire à 500 mm/s s'est déplacé de 10–17 cm. |

### 7.3 Plan d'intégration navigation

#### Phase N-1 : Mise à jour costmap multi-adversaires

| Tâche                               | Fichiers                     | Détail                                                                                         |
| ----------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------- |
| Support N adversaires simultanés    | `Costmap.h/.cpp`             | Tableau `OpponentSlot[MAX_OPPONENTS]` avec position + timestamp, clearing indépendant par slot |
| Nouveau message `MSG_OPPONENTS_POS` | `CommDefs.h`, `Commands.cpp` | Payload variable : `{count: u8, [{x_cm: u16, y_cm: u16}]}` — remplace `MSG_OPPONENT_POS`       |
| Vieillissement par slot             | `Costmap.cpp`, `main.ino`    | Chaque slot a son propre timeout. Un adversaire non vu depuis 2s est effacé individuellement.  |

#### Phase N-2 : Obstacles noisettes sur costmap (optionnel)

| Tâche                                                  | Fichiers      | Détail                                                                                                                        |
| ------------------------------------------------------ | ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Injection des caisses détectées comme obstacles légers | `Costmap.cpp` | `updateNut(x_cm, y_cm)` — disque plus petit (10 cm) avec `NAV_INFLATED` au lieu de `NAV_OBSTACLE` (traversables mais coûteux) |
| Clearing automatique quand une caisse est ramassée     | `Costmap.cpp` | Retirer le blob quand le marqueur disparaît (caisse ramassée = plus visible)                                                  |

#### Phase N-3 : Correction de position par vision (fusion caméra / odométrie)

**Architecture proposée** :

```
                   ┌─────────────────────────┐
  Caméra ArUco ───→│  Position externe       │
  (x_cam, y_cam,  │  (en mm, ~10mm précision,│
   θ_cam)         │   200-350ms de latence)  │
                   └───────┬─────────────────┘
                           │
                           ▼
                   ┌─────────────────────────┐
  Encodeurs ──────→│  Filtre de correction    │──→ state.robot_x/y/theta
  IMU (gyro) ─────→│  (complémentaire ou EKF) │
                   └─────────────────────────┘
```

**Options de fusion** :

| Approche                               | Complexité | Robustesse | Recommandation                     |
| -------------------------------------- | ---------- | ---------- | ---------------------------------- |
| **A. Reset doux périodique**           | Faible     | Moyenne    | ✅ **Recommandé pour Eurobot**     |
| B. Filtre complémentaire (alpha blend) | Moyenne    | Bonne      | Alternative viable                 |
| C. EKF complet (6 états)               | Élevée     | Excellente | Sur-dimensionné pour 100s de match |

**Approche A — Reset doux périodique** (détaillée) :

```
Toutes les N frames (ex: 500 ms) :
  1. Recevoir (x_cam, y_cam, θ_cam) du pipeline vision
  2. Calculer erreur = distance(odom, cam)
  3. Si erreur < SEUIL_REJET (ex: 30 cm) :     // rejeter les outliers
       correction = ALPHA * (cam - odom)         // ALPHA = 0.3–0.5
       state.robot_x += correction.x
       state.robot_y += correction.y
       state.robot_theta += ALPHA_THETA * angularDiff(θ_cam, θ_odom)
  4. Si erreur > SEUIL_REJET : ignorer (faux positif ou ID swappé)
```

| Paramètre               | Valeur suggérée | Justification                                           |
| ----------------------- | --------------- | ------------------------------------------------------- |
| Fréquence de correction | 2–5 Hz          | Plus rapide que la dérive, plus lent que le PID (50 Hz) |
| `ALPHA` (position)      | 0.3             | Convergence en ~5 corrections (~2s) sans à-coup         |
| `ALPHA_THETA` (angle)   | 0.2             | Plus conservateur — l'IMU est déjà fiable pour θ        |
| `SEUIL_REJET`           | 30 cm           | Au-delà, c'est probablement une erreur de détection     |

**Messages firmware nécessaires** :

| Message                           | Direction    | Payload                                                           |
| --------------------------------- | ------------ | ----------------------------------------------------------------- |
| `MSG_VISION_POSE` (nouveau, 0x0A) | Bridge→Robot | `{robot_id: u8, x_mm: u16, y_mm: u16, theta_cdeg: i16}` — 7 bytes |

**Fichiers impactés** :

- `CommDefs.h` : nouveau `MSG_VISION_POSE`
- `Commands.cpp` : handler `onVisionPose()`
- `Odometry.h/.cpp` : `applyVisionCorrection(x_mm, y_mm, theta_rad)`
- `dashboard/apps/server/src/socket.ts` : relayer positions robots alliés vers serial bridge
- `vision/marker_detection/markers.py` : envoyer positions robots alliés en mm + angle

### 7.4 Priorisation navigation

| Priorité | Action                            | Pré-requis                                               | Effort      |
| -------- | --------------------------------- | -------------------------------------------------------- | ----------- |
| **N-1**  | Multi-adversaires costmap         | Classification IDs corrigée + sortie mm                  | Moyen       |
| **N-2**  | Obstacles noisettes costmap       | Classification IDs corrigée + sortie mm                  | Faible      |
| **N-3**  | Correction de position par vision | Envoi positions alliés en mm + nouveau `MSG_VISION_POSE` | Moyen-élevé |

---

## 8. Plan de corrections prioritaires (révisé)

### P0 — Bloquant (sans ça, les objectifs ne sont pas atteignables)

| #   | Action                                                                                          | Fichiers                                                              | Effort  |
| --- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------- |
| 1   | Classifier les IDs 36/47/41 comme `NUT_BLUE`/`NUT_YELLOW`/`NUT_EMPTY` (résout aussi la couleur) | `config.py`, `markers.py`                                             | Trivial |
| 2   | Sortie en mm au lieu de cellules de grille                                                      | `geometry.py`, `markers.py`, `esp32_sender.py`, `dashboard_sender.py` | Faible  |
| 3   | Extraction d'orientation (angle) depuis les coins du marqueur                                   | `markers.py` ou nouveau `pose.py`                                     | Faible  |
| 4   | Extension du protocole série : `TYPE,X_MM,Y_MM,ANGLE_DEG\n`                                     | `esp32_sender.py`, firmware bridge                                    | Faible  |

### P1 — Dashboard (rendre les données exploitables)

| #   | Action                                                                | Fichiers              | Effort |
| --- | --------------------------------------------------------------------- | --------------------- | ------ |
| 5   | Remplacer `DashboardSender` HTTP → Socket.IO client                   | `dashboard_sender.py` | Faible |
| 6   | Récepteur Socket.IO côté server + broadcast `cameraDetections`        | `socket.ts`           | Faible |
| 7   | Afficher noisettes sur `MapCanvas` (rectangles colorés + orientation) | `MapCanvas.tsx`       | Moyen  |
| 8   | Afficher adversaires caméra sur `MapCanvas` (remplacer cercle unique) | `MapCanvas.tsx`       | Faible |
| 9   | Relayer détections adversaires vers robots via serial bridge          | `socket.ts`           | Faible |

### P2 — Navigation (exploiter la vision pour la navigation)

| #   | Action                                                     | Fichiers                                     | Effort |
| --- | ---------------------------------------------------------- | -------------------------------------------- | ------ |
| 10  | Support multi-adversaires sur costmap                      | `Costmap.h/.cpp`, `CommDefs.h`               | Moyen  |
| 11  | Correction de position par vision (reset doux)             | `CommDefs.h`, `Commands.cpp`, `Odometry.cpp` | Moyen  |
| 12  | Envoi position robots alliés via dashboard → bridge serial | `socket.ts`, `CommDefs.h`                    | Moyen  |
| 13  | Obstacles noisettes sur costmap (traversables)             | `Costmap.cpp`                                | Faible |

### P3 — Robustesse

| #   | Action                                   | Fichiers                             | Effort |
| --- | ---------------------------------------- | ------------------------------------ | ------ |
| 14  | Calibration caméra (damier → distorsion) | Nouveau `calibrate.py`, `runtime.py` | Moyen  |
| 15  | Filtre de Kalman par marqueur            | `tracking.py`                        | Moyen  |
| 16  | Reconnexion automatique caméra           | `runtime.py`, `detect_markers.py`    | Faible |
| 17  | Métriques de performance (FPS, latence)  | `detect_markers.py`                  | Faible |

---

## 9. Synthèse

### 9.1 Pipeline vision

| Critère                   | État                                                                                           |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| Détection ArUco robots    | ✅ Fonctionnel (position grille uniquement)                                                    |
| Détection ArUco noisettes | ⚠️ Détecté mais mal classifié (tout en `AREA`)                                                 |
| Couleur des noisettes     | ✅ Déductible de l'ID ArUco (36=bleu, 47=jaune, 41=vide) — nécessite correction classification |
| Position en mm            | ❌ Sortie en cellules de grille (100 mm)                                                       |
| Orientation des noisettes | ❌ Non extrait (coins disponibles mais angle non calculé)                                      |
| Orientation des robots    | ❌ Non extrait                                                                                 |
| Calibration caméra        | ❌ Non implémenté                                                                              |
| Tracking prédictif        | ❌ Moyenne glissante uniquement                                                                |
| Robustesse Iriun          | ⚠️ Probing existant mais pas de reconnexion auto                                               |

### 9.2 Intégration dashboard

| Critère                         | État                                                                 |
| ------------------------------- | -------------------------------------------------------------------- |
| Vision → dashboard              | ❌ Endpoint HTTP inexistant (404), script référencé obsolète         |
| Noisettes sur la carte          | ❌ Affichage texte uniquement, pas sur `MapCanvas`                   |
| Adversaires caméra sur la carte | ⚠️ Cercle unique depuis store, pas alimenté par caméra en production |
| Relay détections → robots       | ❌ Non implémenté (seulement obstacles manuels)                      |

### 9.3 Aide à la navigation

| Critère                        | État                                            |
| ------------------------------ | ----------------------------------------------- |
| Adversaire sur costmap         | ✅ 1 adversaire, disque 20 cm, timeout 2s       |
| Multi-adversaires              | ❌ Un seul slot, écrasement                     |
| Noisettes sur costmap          | ❌ Non implémenté                               |
| Correction position par vision | ❌ Aucun mécanisme (odométrie pure encoder+IMU) |
| Fusion IMU/encodeurs           | ✅ `AngleFusion` filtre complémentaire pour θ   |

### 9.4 Verdict

Le pipeline vision détecte les marqueurs ArUco de manière fiable mais les données sont **sous-exploitées** : mauvaise classification des noisettes, résolution 10× trop faible, aucune intégration dashboard fonctionnelle, et la position vue par la caméra n'est jamais utilisée pour corriger l'odométrie.

**4 corrections P0** (classification IDs, sortie mm, orientation, protocole série) débloquent toute la chaîne. Elles sont toutes de faible effort et sans dépendance externe.

L'intégration dashboard (P1) et la correction de position par vision (P2) sont les gains les plus impactants pour la compétition, mais nécessitent un travail coordonné entre les couches Python, TypeScript et C++.
