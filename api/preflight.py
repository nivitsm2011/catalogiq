"""Startup check: tell the user exactly which files are missing and how to create them."""

from __future__ import annotations

from catalogiq.config import get_api_config, resolve_path


class MissingFilesError(RuntimeError):
    """Raised at startup when required model or data files are absent."""


def missing_files(required: list[dict[str, str]] | None = None) -> list[tuple[str, str]]:
    """Return ``(path, how_to_create_it)`` for every required file that does not exist."""
    items = required if required is not None else get_api_config()["required_files"]
    return [(i["path"], i["how"]) for i in items if not resolve_path(i["path"]).exists()]


def check_required_files(required: list[dict[str, str]] | None = None) -> None:
    """Raise ``MissingFilesError`` with one instruction per missing file.

    Raises:
        MissingFilesError: listing each missing path and the command that creates it.
    """
    missing = missing_files(required)
    if not missing:
        return
    lines = [f"  - {path}\n      -> {how}" for path, how in missing]
    raise MissingFilesError(
        "CatalogIQ cannot start: required model or data files are missing.\n"
        + "\n".join(lines)
        + "\nThe README sections 'Quick start' and 'Data pipeline' list the steps in order."
    )
