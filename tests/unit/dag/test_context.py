from unittest.mock import MagicMock, patch

from restorax.dag.context import ProgressEmitter


def test_progress_emitter_publishes_to_redis():
    with patch("redis.from_url") as mock_from_url:
        mock_redis = MagicMock()
        mock_from_url.return_value = mock_redis

        emitter = ProgressEmitter(job_id="job-123", redis_url="redis://localhost:6379/0")
        emitter.emit(node_id="restore_1", progress=0.5, branch_index=1, status="running")

        mock_redis.publish.assert_called_once()
        channel, payload_str = mock_redis.publish.call_args[0]
        assert channel == "restorax:job_progress:job-123"

        import json

        payload = json.loads(payload_str)
        assert payload["node_id"] == "restore_1"
        assert payload["branch_index"] == 1
        assert payload["progress"] == 0.5
        assert payload["status"] == "running"


def test_progress_emitter_swallows_redis_errors():
    with patch("redis.from_url") as mock_from_url:
        mock_redis = MagicMock()
        mock_redis.publish.side_effect = ConnectionError("redis down")
        mock_from_url.return_value = mock_redis

        emitter = ProgressEmitter(job_id="job-xyz", redis_url="redis://localhost:6379/0")
        emitter.emit("n1", 0.3)  # must not raise
