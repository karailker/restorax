"""Tests for DAG job task integration."""


def test_run_dag_job_task_is_registered():
    from restorax.tasks.job_tasks import run_dag_job
    assert run_dag_job.name == "restorax.tasks.job_tasks.run_dag_job"
