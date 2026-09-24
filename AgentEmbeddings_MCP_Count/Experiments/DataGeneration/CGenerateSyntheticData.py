'''
This class will create synthetic data for testing the agent embeddings model.

and insert it into the database.

All ids start from 1.
Other numbers start from 0

'''
import os,sys
import random
import math
import numpy as np
from tqdm import tqdm

# ----------------------------------------------
# Ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.CConfig import CConfig
from Experiments.Database.CDatabaseManager import CDatabaseManager
from Experiments.DataGeneration.CCache import CCache

class CGenerateSyntheticData:
    def __init__(self):
        self.dbManager = CDatabaseManager()
        self.dbManager.create_tables()
        self.cache = CCache()
        random.seed(42)  # For reproducibility
        self.canary_1_agent_ids = []
        self.canary_2_agent_ids = []
        self.ref_canary_1_agent_id = -1
        self.ref_canary_2_agent_id = -1
        self.ref_canary_1_session_lengths = None
        self.ref_canary_1_session_interactions = None
        self.ref_canary_2_session_lengths = None
        self.ref_canary_2_session_interactions = None
       
        
    def __create_agents__(self):
        print("Creating agents...")
        for agent_id in tqdm(range(1, CConfig.MAX_AGENTS+1)):
            insert_agent_query = "INSERT INTO agents (id) VALUES (?);"
            self.dbManager.execute_query(insert_agent_query, (agent_id,))
            self.cache.add_agent(agent_id)
        print("All Agents created.")

    def __create_canary_agents_tables__(self):
        print("Creating canary agents...")
        # References Canary Agent
        # The reference canary agent will be 1 for each category
        allAgentIds = self.cache.get_all_agent_ids()
        # Randomly pick a reference agent for each canary agent
        # Create a new list of all agents which are not canary agents
        # Calculate number of canary agents
        num_canary_1 = math.ceil((CConfig.PERCENTAGE_AGENTS_CANARY_1 / 100) * len(allAgentIds)) # use allAgentIds and not CConfig
        num_canary_2 = math.ceil((CConfig.PERCENTAGE_AGENTS_CANARY_2 / 100) * len(allAgentIds))
        # Insure no overlap between canary agent ids, they will come from different ranges
        self.canary_1_agent_ids = random.sample(range(1, math.ceil(len(allAgentIds)/2)), num_canary_1+1) # +1 to insure we have 2 Canary agents
        self.canary_2_agent_ids = random.sample(range(math.ceil(len(allAgentIds)/2)+1, len(allAgentIds)), num_canary_2 + 1)
        
        # sort the canary agent ids
        self.canary_1_agent_ids.sort()
        self.canary_2_agent_ids.sort()
        # pick smallest id as reference canary agent
        self.ref_canary_1_agent_id = self.canary_1_agent_ids[0]
        self.ref_canary_2_agent_id = self.canary_2_agent_ids[0]

        # Add the canary agents to the DB
        for agent_id in self.canary_1_agent_ids:
            insert_canary_query = "INSERT INTO canary_agents (agent_id, canary_category) VALUES (?, ?);"
            self.dbManager.execute_query(insert_canary_query, (agent_id, 1))
        for agent_id in self.canary_2_agent_ids:
            insert_canary_query = "INSERT INTO canary_agents (agent_id, canary_category) VALUES (?, ?);"
            self.dbManager.execute_query(insert_canary_query, (agent_id, 2))

        print("All Canary Agents created.")

    def __create_mcp_servers_and_tools__(self):
        print("Creating MCP servers and tools...")
        for mcp_server_id in tqdm(range(1, CConfig.MAX_MCP_SERVERS+1)):
            no_of_tools = random.randint(CConfig.MIN_TOOLS_PER_MCP_SERVER, CConfig.MAX_TOOLS_PER_MCP_SERVER)
            insert_mcp_query = "INSERT INTO mcp_servers (id, no_of_tools) VALUES (?, ?);"
            self.dbManager.execute_query(insert_mcp_query, (mcp_server_id, no_of_tools))
            for mcp_tool_id in range(1, no_of_tools+1):
                insert_tool_query = "INSERT INTO mcp_tools (mcp_server_id, mcp_tool_id) VALUES (?, ?);"
                tool_id = self.dbManager.execute_query(insert_tool_query,(mcp_server_id, mcp_tool_id))
                self.cache.add_mcp_tool(mcp_server_id, mcp_tool_id, tool_id)
        print("MCP servers and tools created.")

    # Before running this, ensure agents and MCP Servers are created
    def __create_sessions_and_interactions_for_agent__(self, agent_id):
        # Agent different handling for canary agents
        if agent_id in self.canary_1_agent_ids and agent_id != self.ref_canary_1_agent_id: # ref has to be created first
            self.__update_sessions_and_interactions_for_canary_agent__(agent_id,
                                                                     self.ref_canary_1_session_lengths,
                                                                     self.ref_canary_1_session_interactions,
                                                                     1)
            return
        if agent_id in self.canary_2_agent_ids and agent_id != self.ref_canary_2_agent_id: # ref. has to be created first
            self.__update_sessions_and_interactions_for_canary_agent__(agent_id,
                                                                     self.ref_canary_2_session_lengths,
                                                                     self.ref_canary_2_session_interactions,
                                                                     2)
            return
            
        
        num_sessions = min(
            CConfig.MAX_SESSIONS_PER_AGENT,
            max(
                1,
                int(
                    np.random.normal(
                        CConfig.SESSIONS_PER_AGENT_MEAN,
                        CConfig.SESSIONS_PER_AGENT_STD,
                    )
                ),
            ),
        )
        for session_index in range(num_sessions):
            session_length = max(1, int(np.random.normal(CConfig.SESSIONS_LENGTH_MEAN, CConfig.SESSIONS_LENGTH_STD)))
            insert_session_query = "INSERT INTO sessions (agent_id, session_depth) VALUES (?, ?);"
            session_id = self.dbManager.execute_query(insert_session_query, (agent_id, session_length))
           
            # Now insert into session_interaction_details
            mcp_server_id = random.randint(1, CConfig.MAX_MCP_SERVERS)
            for seq_num in range(session_length): # Use cache for faster reads
                tool_calls_in_sequence = random.randint(
                    CConfig.MIN_TOOL_CALLS_PER_SEQUENCE,
                    CConfig.MAX_TOOL_CALLS_PER_SEQUENCE,
                )
                if tool_calls_in_sequence == 0:
                    insert_session_interaction_query = "INSERT INTO session_interactions (session_id, tool_id, sequence_number) VALUES (?, ?, ?);"
                    self.dbManager.execute_query(
                        insert_session_interaction_query,
                        (session_id, CConfig.NO_TOOL_CALL_ID, seq_num),
                    )
                    continue
                for _ in range(tool_calls_in_sequence):
                    no_of_tools = self.cache.get_number_of_tools_for_server(mcp_server_id)
                    # Give preference to MCP server used from last prompt
                    mcp_tool_id = random.randint(1, no_of_tools) # CAUTION Use main tool id
                    tool_id = self.cache.get_tool_id(mcp_server_id, mcp_tool_id)
                    insert_session_interaction_query = "INSERT INTO session_interactions (session_id, tool_id, sequence_number) VALUES (?, ?, ?);"
                    self.dbManager.execute_query(insert_session_interaction_query, (session_id, tool_id, seq_num))
                    # ----------- Compute same MCP server with some probability --------------
                    # Only change mcp server if random prob is more than given
                    if random.random() > CConfig.PROB_OF_TOOL_FROM_SAME_MCP: # random.random() gives [0.0, 1.0)
                        mcp_server_id = random.randint(1, CConfig.MAX_MCP_SERVERS)
        # Store reference canary agent sessions and interactions
        # Since agent ids are sorted, this will insure canary agent ref is stored first before
        # another canary agent is created
        if agent_id == self.ref_canary_1_agent_id:
            self.__assign_sessions_and_interactions_for_ref_canary_agents__(1)
        if agent_id == self.ref_canary_2_agent_id:
            self.__assign_sessions_and_interactions_for_ref_canary_agents__(2)
        return


    def __update_sessions_and_interactions_for_canary_agent__(self,
                                                             agent_id,
                                                             ref_session_lengths:dict,
                                                             ref_session_interactions:dict,
                                                             canary_category):
        
        # Delete existing sessions and interactions for this canary agent
        #self.dbManager.delete_all_session_data_for_agent(agent_id)
        
        # Start Updating sessions and interactions
        for session_id in ref_session_lengths.keys():
            session_length = ref_session_lengths[session_id]
            
            if(canary_category == 2):
                # For canary category 2, randomly decrease session length by 1
                if session_length > 1:
                    session_length = session_length - random.randint(0,1)

            insert_session_query = "INSERT INTO sessions (agent_id, session_depth) VALUES (?, ?);"
            new_session_id = self.dbManager.execute_query(insert_session_query, (agent_id, session_length))
           
            # Preserve the reference sequence number for every copied tool call.
            for seq_num, tool_id in ref_session_interactions.get(session_id, []):
                if seq_num >= session_length:
                    break
                insert_session_interaction_query = "INSERT INTO session_interactions (session_id, tool_id, sequence_number) VALUES (?, ?, ?);"
                self.dbManager.execute_query(insert_session_interaction_query, (new_session_id, tool_id, seq_num))
        return
                
    def __create_sessions_and_interactions_for_all_agents__(self):
        print("Creating sessions and interactions for all agents...")
        self.__create_canary_agents_tables__() # This should be done first
        for agent_id in tqdm(range(1, CConfig.MAX_AGENTS + 1)):
            self.__create_sessions_and_interactions_for_agent__(agent_id)

    def __assign_sessions_and_interactions_for_ref_canary_agents__(self, canary_category):
        if canary_category == 1:
            print(f"Get sessions and interactions for canary #1 ref. agent id {self.ref_canary_1_agent_id}  ...")
            # Get session length based on agent id
            self.ref_canary_1_session_lengths = self.dbManager.get_all_session_lengths(self.ref_canary_1_agent_id)
            self.ref_canary_1_session_interactions = self.dbManager.get_session_interactions(self.ref_canary_1_agent_id)

        if canary_category == 2:
            print(f"Get sessions and interactions for canary #2 ref. agent id {self.ref_canary_2_agent_id}  ...")
            # Get session length based on agent id
            self.ref_canary_2_session_lengths = self.dbManager.get_all_session_lengths(self.ref_canary_2_agent_id)
            self.ref_canary_2_session_interactions = self.dbManager.get_session_interactions(self.ref_canary_2_agent_id)

    """
    def __update_sessions_and_interactions_for_all_canary_agents__(self):
        self.dbManager.delete_data_from_tables(['canary_agents'])
        self.__create_canary_agents__()

        # Get all Canary Agents from DB
        allCanaryAgents = self.dbManager.get_canary_agents()

        # Pick 1 canary agent as reference
        ref_canary_1_agent_idx = random.sample(range(0, len(allCanaryAgents[1])),1)[0] # This returns a list so take 1
        ref_canary_2_agent_idx = random.sample(range(0, len(allCanaryAgents[2])),1)[0] # This returns a list so take 1

        ref_canary_1_agent_id = allCanaryAgents[1][ref_canary_1_agent_idx]
        ref_canary_2_agent_id = allCanaryAgents[2][ref_canary_2_agent_idx]

        # Randomly pick a reference agent for each canary agent
        # Create a new list of all agents which are not canary agents
        allCanaryAgents[1].remove(ref_canary_1_agent_id)
        allCanaryAgents[2].remove(ref_canary_2_agent_id)
    
        print(f"Updating sessions and interactions for canary #1 using ref. agent id {ref_canary_1_agent_id}  ...")
        # Get session length based on agent id
        ref_session_lengths = self.dbManager.get_all_session_lengths(ref_canary_1_agent_id)
        ref_session_interactions = self.dbManager.get_session_interactions(ref_canary_1_agent_id)
        
        for idx in tqdm(range(0, len(allCanaryAgents[1]))):
            self.__update_sessions_and_interactions_for_canary_agent__(allCanaryAgents[1][idx],
                                                                      ref_session_lengths,
                                                                      ref_session_interactions,1) 
        
        print(f"Updating sessions and interactions for canary #2 using ref. agent id {ref_canary_2_agent_id}  ...")
        # Get session length based on agent id
        ref_session_lengths = self.dbManager.get_all_session_lengths(ref_canary_2_agent_id)
        ref_session_interactions = self.dbManager.get_session_interactions(ref_canary_2_agent_id)
        
        for idx in tqdm(range(0, len(allCanaryAgents[2]))):
            self.__update_sessions_and_interactions_for_canary_agent__(allCanaryAgents[2][idx],
                                                                      ref_session_lengths,
                                                                      ref_session_interactions,2) 
        return
    """

    def generate_synthetic_data(self):
        dataGenerator = CGenerateSyntheticData()
        # Order is important here
        dataGenerator.__create_agents__()
        dataGenerator.__create_mcp_servers_and_tools__()
        dataGenerator.__create_sessions_and_interactions_for_all_agents__()
        #dataGenerator.__update_sessions_and_interactions_for_all_canary_agents__()

if __name__ == "__main__":
    CDatabaseManager.delete_db_file()
    dataGenerator = CGenerateSyntheticData() # This will create the file so dont move it earlier
    dataGenerator.generate_synthetic_data()
    #dataGenerator.__update_sessions_and_interactions_for_all_canary_agents__()
    