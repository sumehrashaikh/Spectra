"""Light / dark theme support for the SPECTRA desktop GUI.

The theme is applied to the whole Qt application (palette + stylesheet)
and to every embedded Matplotlib canvas, so the plots stay readable when
the UI goes dark.

Light mode is the original look and deliberately applies **no**
stylesheet: switching back restores exactly the previous appearance
instead of approximating it.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

__all__ = [
    "DARK",
    "LIGHT",
    "THEMES",
    "apply_theme",
    "normalize",
    "other_theme",
    "style_canvas",
]

DARK = "dark"
LIGHT = "light"
THEMES = (LIGHT, DARK)

#: Backdrop / surface / text colours used by both Qt and Matplotlib.
DARK_COLORS = {
    "window": "#1e1f22",
    "surface": "#26282c",
    "surface_alt": "#2d3035",
    "text": "#e8e8ea",
    "muted": "#a9adb5",
    "border": "#3a3d43",
    "accent": "#3d7eff",
    "grid": "#3a3d43",
}

LIGHT_COLORS = {
    "window": "#f5f5f7",
    "surface": "#ffffff",
    "surface_alt": "#fbfbfc",
    "text": "#1b1d21",
    "muted": "#5a5f68",
    "border": "#c8cbd1",
    "accent": "#0d6efd",
    "grid": "#cccccc",
}

_DARK_STYLESHEET = """
QWidget {{
    background-color: {window};
    color: {text};
}}
QFrame, QScrollArea, QTabWidget::pane {{
    background-color: {surface};
    border: 1px solid {border};
}}
QLabel {{
    background-color: transparent;
    color: {text};
}}
QPushButton, QToolButton {{
    background-color: {surface_alt};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 10px;
}}
QPushButton:hover, QToolButton:hover {{
    background-color: {border};
}}
QPushButton:disabled {{
    color: {muted};
}}
QLineEdit, QComboBox, QSpinBox {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 2px 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {surface};
    color: {text};
    selection-background-color: {accent};
}}
QTabBar::tab {{
    background-color: {surface_alt};
    color: {muted};
    padding: 5px 10px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background-color: {surface};
    color: {text};
}}
QTableWidget, QHeaderView::section {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border};
    gridline-color: {border};
}}
QCheckBox {{
    background-color: transparent;
}}
QProgressBar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 4px;
}}
QProgressBar::chunk {{
    background-color: {accent};
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {window};
    border: none;
}}
QScrollBar::handle {{
    background: {border};
    border-radius: 4px;
}}
"""


def normalize(mode: str) -> str:
    """Return a canonical theme name (defaults to light)."""
    return DARK if str(mode).strip().lower() == DARK else LIGHT


def other_theme(mode: str) -> str:
    """The theme the toggle button switches to."""
    return DARK if normalize(mode) == LIGHT else LIGHT


def stylesheet_for(mode: str) -> str:
    """Qt stylesheet for ``mode`` (light keeps the original look)."""
    if normalize(mode) != DARK:
        return ""
    return _DARK_STYLESHEET.format(**DARK_COLORS)


def palette_for(mode: str) -> QPalette:
    """Qt palette for ``mode``."""
    colors = DARK_COLORS if normalize(mode) == DARK else LIGHT_COLORS
    palette = QPalette()
    window = QColor(colors["window"])
    surface = QColor(colors["surface"])
    text = QColor(colors["text"])
    muted = QColor(colors["muted"])
    accent = QColor(colors["accent"])

    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, surface)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["surface_alt"]))
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, QColor(colors["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.ToolTipBase, surface)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    return palette


def style_canvas(canvas, mode: str) -> None:
    """Recolour one embedded Matplotlib canvas for ``mode``."""
    if canvas is None:
        return
    colors = DARK_COLORS if normalize(mode) == DARK else LIGHT_COLORS
    try:
        figure = canvas.figure
        figure.set_facecolor(colors["window"])
        for axes in figure.axes:
            axes.set_facecolor(colors["surface"])
            for spine in axes.spines.values():
                spine.set_color(colors["border"])
            axes.tick_params(colors=colors["muted"])
            axes.xaxis.label.set_color(colors["muted"])
            axes.yaxis.label.set_color(colors["muted"])
            axes.title.set_color(colors["text"])
            axes.grid(True, color=colors["grid"], alpha=0.6)
        canvas.draw_idle()
    except Exception:  # noqa: BLE001 - cosmetics must never break analysis
        pass


def apply_theme(widget: QWidget, mode: str, app: QApplication | None = None) -> str:
    """Apply ``mode`` to the application and every canvas under ``widget``.

    Returns the canonical theme name that was applied.
    """
    resolved = normalize(mode)
    application = app or QApplication.instance()

    if application is not None:
        application.setPalette(palette_for(resolved))
        application.setStyleSheet(stylesheet_for(resolved))

    for canvas in _iter_canvases(widget):
        style_canvas(canvas, resolved)

    return resolved


def _iter_canvases(widget: QWidget):
    """Yield every embedded Matplotlib FigureCanvas under ``widget``."""
    if widget is None:
        return
    for child in widget.findChildren(QWidget):
        figure = getattr(child, "figure", None)
        if figure is not None and hasattr(child, "draw_idle"):
            yield child
