import sys
import logging
from PyQt6 import QtWidgets
from .windows.diff_view_window import DiffViewWindow
from . import UI_DIR

def start_diffmap():
    # configure logging…
    print("Creating QApplication...")
    app = QtWidgets.QApplication(sys.argv)

    # Apply stylesheet if exists
    qss_file = UI_DIR / "css" / "uswds.qss"
    if qss_file.exists():
        with open(qss_file, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    print("Creating DiffViewWindow...")
    try:
        win = DiffViewWindow()
        print("Showing window...")
        win.show()
        print("Starting event loop...")
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error creating or showing window: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    start_diffmap()
