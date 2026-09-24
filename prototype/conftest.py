import os
import sys
from pathlib import Path

# Add the parent directory of prototype to PYTHONPATH
# so that imports like `from prototype.core...` work in tests.
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
