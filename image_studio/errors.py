"""Application errors that are independent of the presentation layer."""


class AppError(Exception):
    """Base class for errors that may safely cross an application boundary."""


class UserInputError(AppError):
    """The request is invalid and should be shown to the user verbatim."""


class BackendUnavailableError(AppError):
    """A required local or remote backend cannot currently be used."""


class ModelLoadError(AppError):
    """A model or one of its assets could not be loaded."""


class StorageError(AppError):
    """An output or metadata artifact could not be safely persisted."""
