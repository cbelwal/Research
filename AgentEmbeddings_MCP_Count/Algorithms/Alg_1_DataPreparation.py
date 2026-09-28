"""
Code to run Algorithm 1 (Data Preparation) given in paper

Variables names have been kept similar to those in the paper for ease of understanding.

The algorithms has dependencies on other Classes which are defined in different files.
"""
import os,sys
import time
import torch
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from tqdm import tqdm
from Experiments.CConfig import CConfig
from Experiments.Database.CDataPreparationHelper import CDataPreparationHelper

"""
Inputs: 
Agent and session interaction data from the database

Outputs: 
All_C_hat_a : Dictionary of dictionaries to represent a sparse matrix containing normalized tool call frequencies for each agent
"""
def Algorithm_1_DataPreparation():
    dataPrepHelper = CDataPreparationHelper()
    all_C_hat_a_1 = {} # Dictionary of dictionaries to hold C_hat_a_1 for all agents
    allAgentsIds = dataPrepHelper.get_all_agent_ids()
    for idx in tqdm(range(0,len(allAgentsIds))):
        agentId = allAgentsIds[idx]
        C_a = {}
        C_hat_a = {}
        C_hat_a_1 = {}
        T_a = []
        sessionIdsForAgent = dataPrepHelper.get_sessions_for_agent(agentId)
        total_tool_calls = 0
        for sessionId in sessionIdsForAgent:
            toolIds = dataPrepHelper.get_tools_for_session(sessionId)
            for toolId in toolIds:
                total_tool_calls += 1
                if toolId in C_a:
                    C_a[toolId] += 1
                else:
                    C_a[toolId] = 1
                if toolId not in T_a:
                    T_a.append(toolId)
        # Normalize tool calls by number of sessions
        Sum_C_hat_a = 0
        for toolId in T_a:
            C_hat_a[toolId] = C_a[toolId] / total_tool_calls # len(sessionIdsForAgent)
            Sum_C_hat_a += C_hat_a[toolId]
        
        # Normalize so that value is between 0 and 1
        # so that way we can apply sigmoid
        # C_hat_a_1 is the normalized values on 1.0
        for toolId in T_a:
            C_hat_a_1[toolId] = C_hat_a[toolId] / Sum_C_hat_a

        all_C_hat_a_1[agentId] = C_hat_a_1
    return all_C_hat_a_1


def BuildAgentToolMatrix(
    cache_path: str = None,
    cache_fingerprint: str = None,
    force: bool = False,
):
    if not force and cache_path is not None and os.path.isfile(cache_path):
        print(f"Loading cached agent-tool matrix from {cache_path}")
        try:
            cached = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            cached = torch.load(cache_path, map_location="cpu")
        if (
            isinstance(cached, dict)
            and cached.get("fingerprint") == cache_fingerprint
            and isinstance(cached.get("matrix"), torch.Tensor)
        ):
            return cached["matrix"]
        print("Cached matrix does not match the current database; rebuilding.")

    helper = CDataPreparationHelper(load_session_cache=False)
    db_manager = helper.dbManager
    number_of_agents = len(db_manager.get_all_agent_ids())
    number_of_tools = db_manager.get_number_of_tools()
    matrix = torch.full(
        (number_of_agents, number_of_tools),
        1.0e-4,
        dtype=torch.float32,
    )

    print(
        f"Aggregating tool frequencies for {number_of_agents} agents and "
        f"{number_of_tools} tools..."
    )
    connection = db_manager.sqlLite.conn
    connection.execute("PRAGMA temp_store = FILE;")
    started_at = time.monotonic()
    last_report = [started_at]

    def report_progress():
        now = time.monotonic()
        if now - last_report[0] >= 60:
            print(
                f"Database aggregation still running "
                f"({(now - started_at) / 60:.0f} minutes elapsed)...",
                flush=True,
            )
            last_report[0] = now
        return 0

    connection.set_progress_handler(report_progress, 1000000)
    cursor = connection.cursor()
    cursor.execute(
        """
        WITH tool_counts AS (
            SELECT s.agent_id, si.tool_id, COUNT(*) AS call_count
            FROM session_interactions AS si
            JOIN sessions AS s ON s.id = si.session_id
            WHERE si.tool_id != ?
            GROUP BY s.agent_id, si.tool_id
        )
        SELECT
            agent_id,
            tool_id,
            CAST(call_count AS REAL)
                / SUM(call_count) OVER (PARTITION BY agent_id) AS frequency
        FROM tool_counts
        ORDER BY agent_id, tool_id;
        """,
        (CConfig.NO_TOOL_CALL_ID,),
    )
    connection.set_progress_handler(None, 0)

    while True:
        rows = cursor.fetchmany(100000)
        if not rows:
            break
        agent_indices = torch.tensor(
            [row[0] - 1 for row in rows],
            dtype=torch.long,
        )
        tool_indices = torch.tensor(
            [row[1] - 1 for row in rows],
            dtype=torch.long,
        )
        values = torch.tensor(
            [row[2] for row in rows],
            dtype=torch.float32,
        )
        matrix[agent_indices, tool_indices] = values

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(
            {
                "fingerprint": cache_fingerprint,
                "matrix": matrix,
            },
            cache_path,
        )
        print(f"Cached agent-tool matrix at {cache_path}")
    return matrix
  