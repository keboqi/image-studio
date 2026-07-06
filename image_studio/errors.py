"""Application errors that are independent of the presentation layer."""


class AppError(Exception):
    """Base class for errors that may safely cross an application boundary."""


class UserInputError(AppError):
    """The request is invalid and should be shown to the user verbatim."""


class BackendUnavailableError(AppError):
    """A required local or remote backend cannot currently be used."""


class BackendBusyError(AppError):
    """A backend cannot be changed while requests are active."""


class ModelNotFoundError(UserInputError):
    """A requested model ID or compatibility alias is not registered."""


class ModelLoadError(AppError):
    """A model or one of its assets could not be loaded."""


class StorageError(AppError):
    """An output or metadata artifact could not be safely persisted."""
