import unittest

from src.models import AgentSpec
from src.simulator import AbortRestartSimulator


class SimulatorTests(unittest.TestCase):
    def test_release_at_completion_does_not_abort(self):
        agents = [
            AgentSpec(1, 1, 20, 0, 0, 5, 0),
            AgentSpec(2, 2, 5, 5, 0, 1, 0),
        ]
        result = AbortRestartSimulator(agents).run(10, 20)
        low_job = next(job for job in result.jobs if job.agent.agent_id == 1)
        self.assertEqual(low_job.completion_time, 5)
        self.assertEqual(low_job.abort_count, 0)

    def test_commit_phase_abort_counts_full_consumed_work(self):
        scale = 10
        agents = [
            AgentSpec(1, 1, 400, 60, 5, 30, 10),
            AgentSpec(2, 2, 200, 0, 3, 20, 5),
            AgentSpec(3, 3, 100, 0, 2, 10, 1),
        ]
        result = AbortRestartSimulator(agents).run(400, 500)
        low_job = next(job for job in result.jobs if job.agent.agent_id == 1)
        self.assertEqual(low_job.abort_count, 1)
        self.assertEqual(low_job.wasted_processing, 40)
        self.assertEqual(low_job.wasted_commit, 5)
        self.assertEqual(low_job.response_time, 98)

    def test_abort_counterexample_misses_deadline(self):
        agents = [
            AgentSpec(1, 1, 200, 0, 5, 30, 10),
            AgentSpec(2, 2, 80, 0, 3, 20, 5),
            AgentSpec(3, 3, 50, 0, 2, 10, 1),
        ]
        result = AbortRestartSimulator(agents).run(200, 250)
        self.assertFalse(result.schedulable)
        self.assertEqual(result.missed_job.agent.agent_id, 1)
        self.assertEqual(result.end_time, 200)


if __name__ == "__main__":
    unittest.main()

