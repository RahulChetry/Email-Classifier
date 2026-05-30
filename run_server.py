from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "server.log"
ERR_PATH = ROOT / "server.err"


if __name__ == "__main__":
    sys.stdout = LOG_PATH.open("a", encoding="utf-8")
    sys.stderr = ERR_PATH.open("a", encoding="utf-8")
    try:
        from app import app, load_or_train_model

        load_or_train_model()
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
