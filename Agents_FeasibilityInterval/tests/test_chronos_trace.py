import unittest

from src.chronos_trace import TraceRequest, replay_requests


class ChronosTraceReplayTests(unittest.TestCase):
    def test_high_priority_release_aborts_low_priority_request(self):
        requests = [
            TraceRequest(1, "code", 0, 1, 1, 1, 1, 8, 1, 100),
            TraceRequest(2, "conversation", 5, 1, 1, 2, 1, 1, 1, 100),
        ]
        result = replay_requests(requests, "abort_restart")
        low = result.jobs[0]
        self.assertEqual(low.aborts, 1)
        self.assertEqual(low.wasted_us, 5)
        self.assertEqual(low.completion_us, 18)

    def test_resume_preserves_progress(self):
        requests = [
            TraceRequest(1, "code", 0, 1, 1, 1, 1, 8, 1, 100),
            TraceRequest(2, "conversation", 5, 1, 1, 2, 1, 1, 1, 100),
        ]
        result = replay_requests(requests, "resume")
        low = result.jobs[0]
        self.assertEqual(low.preemptions, 1)
        self.assertEqual(low.aborts, 0)
        self.assertEqual(low.wasted_us, 0)
        self.assertEqual(low.completion_us, 13)


if __name__ == "__main__":
    unittest.main()
