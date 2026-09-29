"""Minimal .env editing, so a value tuned at runtime survives the next launch."""

from pathlib import Path


def update_env_value(path: Path, key: str, value: str) -> None:
    """Set `key=value` in the .env file at `path`, replacing an existing
    assignment in place (keeping the rest of the file, comments included)
    or appending one. Creates the file if it doesn't exist."""
    lines = path.read_text().splitlines() if path.exists() else []
    assignment = f"{key}={value}"
    for index, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == key and not line.lstrip().startswith("#"):
            lines[index] = assignment
            break
    else:
        lines.append(assignment)
    path.write_text("\n".join(lines) + "\n")
