"""Unit tests for custom exceptions."""

from __future__ import annotations

import pytest

from restorax.core.exceptions import (
    AudioReadError,
    AudioWriteError,
    JobNotFoundError,
    PipelineConfigError,
    RestoraXError,
    RestorerLoadError,
    RestorerNotFoundError,
    StorageError,
    VideoReadError,
    VideoWriteError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        for exc_class in [
            RestorerNotFoundError,
            RestorerLoadError,
            VideoReadError,
            VideoWriteError,
            JobNotFoundError,
            PipelineConfigError,
            StorageError,
            AudioReadError,
            AudioWriteError,
        ]:
            assert issubclass(exc_class, RestoraXError)

    def test_base_inherits_from_exception(self):
        assert issubclass(RestoraXError, Exception)


class TestExceptionRaise:
    @pytest.mark.parametrize(
        "exc_class",
        [
            RestorerNotFoundError,
            RestorerLoadError,
            VideoReadError,
            VideoWriteError,
            JobNotFoundError,
            PipelineConfigError,
            StorageError,
            AudioReadError,
            AudioWriteError,
        ],
    )
    def test_can_raise_and_catch_as_base(self, exc_class):
        with pytest.raises(RestoraXError):
            raise exc_class("test message")

    def test_message_preserved(self):
        exc = RestorerLoadError("weights not found at /models/x.pth")
        assert "weights not found" in str(exc)

    def test_can_catch_specific_type(self):
        with pytest.raises(JobNotFoundError):
            raise JobNotFoundError("job abc not found")
