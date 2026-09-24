import os
import tempfile
import unittest
from unittest.mock import patch

from Experiments.CConfig import CConfig
from Experiments.Database.CDataPreparationHelper import CDataPreparationHelper
from Experiments.Database.CDatabaseManager import CDatabaseManager


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.path_patch = patch.object(
            CDatabaseManager,
            "get_database_file_path",
            return_value=self.db_path,
        )
        self.path_patch.start()
        self.db_manager = CDatabaseManager()
        self.db_manager.create_tables()

    def tearDown(self):
        self.db_manager.sqlLite.conn.close()
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_schema_ensures_referential_sentinel_and_positive_tool_ids(self):
        sentinel = self.db_manager.execute_read_query(
            """
            SELECT id, mcp_server_id, mcp_tool_id
            FROM mcp_tools
            WHERE id = ?;
            """,
            (CConfig.NO_TOOL_CALL_ID,),
        )
        self.assertEqual(sentinel, [(CConfig.NO_TOOL_CALL_ID, None, None)])

        self.db_manager.execute_query(
            "INSERT INTO mcp_servers (id, no_of_tools) VALUES (?, ?);",
            (1, 1),
        )
        tool_id = self.db_manager.execute_query(
            "INSERT INTO mcp_tools (mcp_server_id, mcp_tool_id) VALUES (?, ?);",
            (1, 1),
        )
        self.assertEqual(tool_id, 1)

        foreign_keys = self.db_manager.execute_read_query(
            "PRAGMA foreign_key_list(session_interactions);"
        )
        self.assertIn("mcp_tools", [row[2] for row in foreign_keys])
        self.assertEqual(
            self.db_manager.execute_read_query("PRAGMA foreign_key_check;"),
            [],
        )

    def test_tool_and_matrix_queries_exclude_no_call_sentinel(self):
        self.db_manager.execute_query(
            "INSERT INTO mcp_servers (id, no_of_tools) VALUES (?, ?);",
            (1, 2),
        )
        first_tool_id = self.db_manager.execute_query(
            "INSERT INTO mcp_tools (mcp_server_id, mcp_tool_id) VALUES (?, ?);",
            (1, 1),
        )
        second_tool_id = self.db_manager.execute_query(
            "INSERT INTO mcp_tools (mcp_server_id, mcp_tool_id) VALUES (?, ?);",
            (1, 2),
        )
        self.db_manager.execute_query("INSERT INTO agents (id) VALUES (?);", (1,))
        sentinel_only_session = self.db_manager.execute_query(
            "INSERT INTO sessions (agent_id, session_depth) VALUES (?, ?);",
            (1, 1),
        )
        mixed_session = self.db_manager.execute_query(
            "INSERT INTO sessions (agent_id, session_depth) VALUES (?, ?);",
            (1, 1),
        )
        self.db_manager.execute_query(
            """
            INSERT INTO session_interactions
                (session_id, tool_id, sequence_number)
            VALUES (?, ?, ?);
            """,
            (sentinel_only_session, CConfig.NO_TOOL_CALL_ID, 0),
        )
        self.db_manager.execute_query(
            """
            INSERT INTO session_interactions
                (session_id, tool_id, sequence_number)
            VALUES (?, ?, ?);
            """,
            (mixed_session, CConfig.NO_TOOL_CALL_ID, 0),
        )
        self.db_manager.execute_query(
            """
            INSERT INTO session_interactions
                (session_id, tool_id, sequence_number)
            VALUES (?, ?, ?);
            """,
            (mixed_session, first_tool_id, 0),
        )

        self.assertEqual(self.db_manager.get_number_of_tools(), 2)
        self.assertEqual(
            self.db_manager.get_all_tool_ids(),
            [first_tool_id, second_tool_id],
        )
        self.assertEqual(
            self.db_manager.get_tools_for_session(sentinel_only_session),
            [],
        )
        self.assertEqual(
            self.db_manager.get_tools_for_session(mixed_session),
            [first_tool_id],
        )
        self.assertEqual(
            self.db_manager.get_all_tools_and_sessions(),
            {mixed_session: [first_tool_id]},
        )
        self.assertEqual(
            self.db_manager.get_tool_call_count(CConfig.NO_TOOL_CALL_ID),
            0,
        )
        self.assertEqual(self.db_manager.get_tool_call_count(first_tool_id), 1)

        helper = CDataPreparationHelper.__new__(CDataPreparationHelper)
        helper.all_sessions_data_cache = (
            self.db_manager.get_all_tools_and_sessions()
        )
        self.assertEqual(helper.get_tools_for_session(sentinel_only_session), [])


if __name__ == "__main__":
    unittest.main()
