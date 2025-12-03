# config.py - Relative directory configurations for the project
import os

# Project root directory (resolved dynamically when imported)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Dictionary of key paths (relative to ROOT_DIR)
PATHS = {
    'SCRIPTS_PYTHON': os.path.join(ROOT_DIR, 'scripts', 'python'),
    'GRAPHICS': os.path.join(ROOT_DIR, 'graphics'),
    'UI_BAR': os.path.join(ROOT_DIR, 'graphics', 'ui_bar'),
    'NUMBERS': os.path.join(ROOT_DIR, 'graphics', 'numbers'),
    'BUTTONS': os.path.join(ROOT_DIR, 'graphics', 'buttons'),
    'SOUNDCLIPS': os.path.join(ROOT_DIR, 'soundclips'),
    'BASE': os.path.join(ROOT_DIR, 'photogrammetry'),
    'INIT_PHOTOS': os.path.join(ROOT_DIR, 'graphics', 'initialization'),
}

# Example usage in a script: from config import PATHS; graphics_dir = PATHS['GRAPHICS']