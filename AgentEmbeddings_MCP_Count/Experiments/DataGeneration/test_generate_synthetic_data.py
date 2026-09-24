import unittest
from unittest.mock import patch

from Experiments.CConfig import CConfig
from Experiments.DataGeneration.CGenerateSyntheticData import CGenerateSyntheticData


class _FakeDatabaseManager:
    def __init__(self):
        self.queries = []

    def execute_query(self, query, params=()):
        self.queries.append((query, params))
        if "INSERT INTO sessions" in query:
            return 101
        return len(self.queries)


class _FakeCache:
    def get_number_of_tools_for_server(self, mcp_server_id):
        return 10

    def get_tool_id(self, mcp_server_id, mcp_tool_id):
        return (mcp_server_id * 100) + mcp_tool_id


class TestGenerateSyntheticData(unittest.TestCase):
    def _create_generator(self):
        generator = CGenerateSyntheticData.__new__(CGenerateSyntheticData)
        generator.dbManager = _FakeDatabaseManager()
        generator.cache = _FakeCache()
        generator.canary_1_agent_ids = []
        generator.canary_2_agent_ids = []
        generator.ref_canary_1_agent_id = -1
        generator.ref_canary_2_agent_id = -1
        return generator

    @patch(
        "Experiments.DataGeneration.CGenerateSyntheticData.np.random.normal",
        side_effect=[1, 1],
    )
    @patch(
        "Experiments.DataGeneration.CGenerateSyntheticData.random.random",
        return_value=0.0,
    )
    @patch(
        "Experiments.DataGeneration.CGenerateSyntheticData.random.randint",
        side_effect=[1, 3, 1, 2, 3],
    )
    def test_multiple_tool_calls_share_sequence_number(
        self,
        _mock_randint,
        _mock_random,
        _mock_normal,
    ):
        generator = self._create_generator()

        generator.__create_sessions_and_interactions_for_agent__(1)

        interactions = [
            params
            for query, params in generator.dbManager.queries
            if "INSERT INTO session_interactions" in query
        ]
        self.assertEqual(
            interactions,
            [
                (101, 101, 0),
                (101, 102, 0),
                (101, 103, 0),
            ],
        )

    @patch(
        "Experiments.DataGeneration.CGenerateSyntheticData.np.random.normal",
        side_effect=[1, 1],
    )
    @patch(
        "Experiments.DataGeneration.CGenerateSyntheticData.random.randint",
        side_effect=[1, 0],
    )
    def test_sequence_with_no_tool_calls_inserts_sentinel(
        self,
        _mock_randint,
        _mock_normal,
    ):
        generator = self._create_generator()

        generator.__create_sessions_and_interactions_for_agent__(1)

        interactions = [
            params
            for query, params in generator.dbManager.queries
            if "INSERT INTO session_interactions" in query
        ]
        self.assertEqual(interactions, [(101, CConfig.NO_TOOL_CALL_ID, 0)])

    def test_canary_copy_preserves_sentinel_and_shared_sequence_numbers(self):
        generator = self._create_generator()

        generator.__update_sessions_and_interactions_for_canary_agent__(
            agent_id=2,
            ref_session_lengths={7: 2},
            ref_session_interactions={
                7: [
                    (0, CConfig.NO_TOOL_CALL_ID),
                    (1, 101),
                    (1, 102),
                ],
            },
            canary_category=1,
        )

        interactions = [
            params
            for query, params in generator.dbManager.queries
            if "INSERT INTO session_interactions" in query
        ]
        self.assertEqual(
            interactions,
            [
                (101, CConfig.NO_TOOL_CALL_ID, 0),
                (101, 101, 1),
                (101, 102, 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
