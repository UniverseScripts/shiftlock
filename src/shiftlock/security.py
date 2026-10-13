"""Secret handling, log redaction and private file output."""

import logging
import os
import re
from pathlib import Path

from pydantic import SecretStr

from shiftlock.errors import MissingKeyError

API_KEY_VARIABLES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)(x-api-key|api[_-]?key)(['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9._\-]+"),
]

QUIET_LOGGERS = ("httpx2", "httpcore2", "httpx", "httpcore", "openai", "anthropic")

WORKSPACE = Path(".shiftlock")


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(_mask, text)
    return text


def _mask(match: re.Match[str]) -> str:
    if match.lastindex and match.lastindex >= 2:
        return f"{match.group(1)}{match.group(2)}[REDACTED]"
    return "[REDACTED]"


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


def configure_logging(verbose: bool, debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def read_api_key(provider: str) -> SecretStr:
    variable = API_KEY_VARIABLES[provider]
    value = os.environ.get(variable, "").strip()
    if not value:
        raise MissingKeyError(
            f"{variable} is not set. Export it in your shell; keys are never read from arguments or files."
        )
    return SecretStr(value)


def make_private_directory(path: Path) -> Path:
    missing = [directory for directory in (path, *path.parents) if not directory.exists()]
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        directory.chmod(0o700)
    path.chmod(0o700)
    return path


def write_private_file(path: Path, content: str) -> None:
    make_private_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
        file.write(content)
    path.chmod(0o600)


def ensure_workspace(root: Path = WORKSPACE) -> Path:
    make_private_directory(root)
    ignore_file = root / ".gitignore"
    if not ignore_file.exists():
        write_private_file(ignore_file, "*\n")
    return root
