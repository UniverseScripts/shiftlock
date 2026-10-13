from enum import IntEnum


class ExitCode(IntEnum):
    GO = 0
    NO_GO = 1
    CONDITIONAL = 2
    INCONCLUSIVE = 3
    COST_CAP = 4
    CONFIG_ERROR = 10
    IMPORT_ERROR = 11
    AUTH_ERROR = 12
    PROVIDER_ERROR = 13
    UNEXPECTED_ERROR = 20


class ShiftlockError(Exception):
    exit_code = ExitCode.UNEXPECTED_ERROR


class ConfigError(ShiftlockError):
    exit_code = ExitCode.CONFIG_ERROR


class SuiteImportError(ShiftlockError):
    exit_code = ExitCode.IMPORT_ERROR


class MissingKeyError(ShiftlockError):
    exit_code = ExitCode.AUTH_ERROR


class ProviderError(ShiftlockError):
    exit_code = ExitCode.PROVIDER_ERROR

    def __init__(self, message: str, *, status_code: int | None, retryable: bool, retry_after: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after

    @property
    def is_auth_failure(self) -> bool:
        return self.status_code in (401, 403)
