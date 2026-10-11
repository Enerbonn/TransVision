"""Cooperative cancellation shared by editor and unattended workers."""

import threading
from contextlib import contextmanager


class JobCancelled(Exception):
    """An intentional stop, rather than an export failure."""


_local = threading.local()


def check_cancel(folder=None):
    callback = getattr(_local, "cancelled", None)
    if callback and callback():
        if folder is not None:
            from pathlib import Path

            (Path(folder) / "INCOMPLETE.txt").write_text(
                "Stopped by user. This export may be incomplete; retry to create a fresh export.\n",
                encoding="utf-8",
            )
        raise JobCancelled("Stopped by user. Completed exports remain intact.")


@contextmanager
def cancellable(callback):
    previous = getattr(_local, "cancelled", None)
    _local.cancelled = callback
    try:
        check_cancel()
        yield
    finally:
        _local.cancelled = previous


def cancellation_callback():
    return getattr(_local, "cancelled", None) or (lambda: False)
