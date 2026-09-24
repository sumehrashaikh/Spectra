"""Launch the Spectra desktop GUI.

Entry points:

    python -m prototype.main     # from the repository root
    python main.py               # from the prototype/ folder
    spectra gui                  # installed console script

The GUI is the intended interface for non-terminal users; see
``docs/GUI_USER_GUIDE.md`` for a walkthrough.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from prototype.gui.window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())