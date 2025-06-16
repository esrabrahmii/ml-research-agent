"""Make src/ importable for pytest without an editable install hop."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
