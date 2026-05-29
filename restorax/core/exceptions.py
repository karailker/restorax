class RestoraXError(Exception):
    """Base exception for all RestoraX errors."""


class RestorerNotFoundError(RestoraXError):
    """Raised when a requested restorer name is not in the registry."""


class RestorerLoadError(RestoraXError):
    """Raised when a restorer fails to load its weights."""


class VideoReadError(RestoraXError):
    """Raised on unrecoverable video decoding errors."""


class VideoWriteError(RestoraXError):
    """Raised on unrecoverable video encoding errors."""


class JobNotFoundError(RestoraXError):
    """Raised when a job ID does not exist in the database."""


class PipelineConfigError(RestoraXError):
    """Raised when a pipeline YAML config is invalid."""


class StorageError(RestoraXError):
    """Raised on storage backend read/write failures."""


class AudioReadError(RestoraXError):
    """Raised when audio extraction from a container fails."""


class AudioWriteError(RestoraXError):
    """Raised when audio encoding or muxing into a container fails."""


class DAGValidationError(RestoraXError):
    """Raised when a DAG fails structural validation (cycles, unknown ports, type mismatches)."""


class NodeExecutionError(RestoraXError):
    """Raised when a node's execute() fails after all retries are exhausted."""

    def __init__(self, node_id: str, attempt: int, cause: Exception) -> None:
        super().__init__(f"Node '{node_id}' failed on attempt {attempt}: {cause}")
        self.node_id = node_id
        self.attempt = attempt
        self.__cause__ = cause
        self.__suppress_context__ = True


class PortTypeMismatchError(DAGValidationError):
    """Raised when an edge connects ports with incompatible type hints."""
