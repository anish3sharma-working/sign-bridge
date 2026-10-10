"""Launch the SignBridge Windows desktop application."""
from __future__ import annotations
import logging
import sys
from config.settings import app_data_dir


def main() -> int:
    log_dir = app_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / "signbridge.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from ui.main_window import SignBridgeWindow
    except ImportError as exc:
        print("SignBridge desktop dependencies are missing. Follow the Python 3.14 installation steps in README.md.", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        return 2
    app = QApplication(sys.argv)
    app.setApplicationName("SignBridge")
    app.setOrganizationName("SignBridge Community")
    try:
        window = SignBridgeWindow()
        window.show()
    except Exception as exc:
        logging.exception("Application startup failed")
        QMessageBox.critical(None, "SignBridge could not start", f"A recoverable startup error occurred. Details are in the local log file.\n\n{exc}")
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
