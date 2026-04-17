"""Configuration pytest pour la suite vision.

Ajoute le dossier parent au ``sys.path`` afin que les tests puissent
importer ``marker_detection`` sans installation package.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # programs/vision
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
