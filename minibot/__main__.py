"""`python -m minibot`.

The version check runs BEFORE the first import of the package, and that
ordering is the whole point. On an older interpreter the failure otherwise
surfaces as `TypeError: unsupported operand type(s) for |` from somewhere deep
in the import chain — a message that says nothing about the actual problem,
which is almost always `python3` picking up the system interpreter instead of
the virtualenv.
"""

import sys

MINIMUM = (3, 11)

if sys.version_info < MINIMUM:
    have = ".".join(str(n) for n in sys.version_info[:3])
    need = ".".join(str(n) for n in MINIMUM)
    sys.exit(
        f"minibot needs Python {need} or newer, but this is Python {have}\n"
        f"  ({sys.executable})\n\n"
        "On macOS, `python3` is usually the system interpreter, which is too "
        "old.\nUse the virtualenv instead:\n\n"
        "    .venv/bin/python -m minibot\n\n"
        "If there isn't one yet:\n\n"
        "    python3 -m venv .venv\n"
        "    .venv/bin/pip install -r requirements.txt"
    )

import os  # noqa: E402

# The embedding and Whisper models are loaded through Hugging Face, which draws
# download progress bars even when everything is already cached — a dozen
# lines of "Fetching files / Download complete" on every start, burying the
# log lines that matter. A real first-time download still happens; it is just
# not animated. Set before any import that might pull the hub in.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from .cli import main  # noqa: E402  (must not import before the check above)

if __name__ == "__main__":
    main()
