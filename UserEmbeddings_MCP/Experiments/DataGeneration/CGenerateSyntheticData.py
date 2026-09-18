'''
This class will create synthetic data for testing the user embeddings model.

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
        self.canary_1_user_ids = []
        self.canary_2_user_ids = []
        self.ref_canary_1_user_id = -1
        self.ref_canary_2_user_id = -1
        self.ref_canary_1_session_lengths = None
        self.ref_canary_1_session_interactions = None
        self.ref_canary_2_session_lengths = None
        self.ref_canary_2_session_interactions = None
       
        
    def __create_users__(self):
        print("Creating users...")
        for user_id in tqdm(range(1, CConfig.MAX_USERS+1)):
            insert_user_query = "INSERT INTO users (id) VALUES (?);"
            self.dbManager.execute_query(insert_user_query, (user_id,))
            self.cache.add_user(user_id)
        print("All Users created.")

    def __create_canary_users_tables__(self):
        print("Creating canary users...")
        # References Canary User
        # The reference canary user will be 1 for each category
        allUserIds = self.cache.get_all_user_ids()
        # Randomly pick a reference user for each canary user
        # Create a new list of all users which are not canary users    
        # Calculate number of canary users
        num_canary_1 = math.ceil((CConfig.PERCENTAGE_USERS_CANARY_1 / 100) * len(allUserIds)) # use allUSerIds and not CConfig
        num_canary_2 = math.ceil((CConfig.PERCENTAGE_USERS_CANARY_2 / 100) * len(allUserIds))
        # Insure no overlap between canary user ids, they will come from different ranges
        self.canary_1_user_ids = random.sample(range(1, math.ceil(len(allUserIds)/2)), num_canary_1+1) # +1 to insure we have 2 Canary users
        self.canary_2_user_ids = random.sample(range(math.ceil(len(allUserIds)/2)+1, len(allUserIds)), num_canary_2 + 1)    
        
        # sort the canary user ids
        self.canary_1_user_ids.sort()
        self.canary_2_user_ids.sort()
        # pick smallest id as reference canary user
        self.ref_canary_1_user_id = self.canary_1_user_ids[0]
        self.ref_canary_2_user_id = self.canary_2_user_ids[0]

        # Add the canary users to the DB  
        for user_id in self.canary_1_user_ids:
            insert_canary_query = "INSERT INTO canary_users (user_id, canary_category) VALUES (?, ?);"
            self.dbManager.execute_query(insert_canary_query, (user_id, 1))
        for user_id in self.canary_2_user_ids:
            insert_canary_query = "INSERT INTO canary_users (user_id, canary_category) VALUES (?, ?);"
            self.dbManager.execute_query(insert_canary_query, (user_id, 2))

        print("All Canary Users created.")

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

    # Before running this, ensure users and MCP Servers are created
    def __create_sessions_and_interactions_for_user__(self, user_id):
        # User different handling for canary users
        if user_id in self.canary_1_user_ids and user_id != self.ref_canary_1_user_id: # ref has to be created first
            self.__update_sessions_and_interactions_for_canary_user__(user_id,
                                                                     self.ref_canary_1_session_lengths,
                                                                     self.ref_canary_1_session_interactions,
                                                                     1)
            return
        if user_id in self.canary_2_user_ids and user_id != self.ref_canary_2_user_id: # ref. has to be created first
            self.__update_sessions_and_interactions_for_canary_user__(user_id,
                                                                     self.ref_canary_2_session_lengths,
                                                                     self.ref_canary_2_session_interactions,
                                                                     2)
            return
            
        
        num_sessions = max(1, int(np.random.normal(CConfig.SESSIONS_PER_USER_MEAN, CConfig.SESSIONS_PER_USER_STD)))
        for session_index in range(num_sessions):
            session_length = max(1, int(np.random.normal(CConfig.SESSIONS_LENGTH_MEAN, CConfig.SESSIONS_LENGTH_STD)))
            insert_session_query = "INSERT INTO sessions (user_id, session_depth) VALUES (?, ?);"
            session_id = self.dbManager.execute_query(insert_session_query, (user_id, session_length))
           
            # Now insert into session_interaction_details
            mcp_server_id = random.randint(1, CConfig.MAX_MCP_SERVERS)
            for seq_num in range(session_length): # Use cache for faster reads
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
        # Store reference canary user sessions and interactions
        # Since user ids are sorted, this will insure canary user ref is stored first before
        # another canary user is created
        if user_id == self.ref_canary_1_user_id:
            self.__assign_sessions_and_interactions_for_ref_canary_users__(1)
        if user_id == self.ref_canary_2_user_id:
            self.__assign_sessions_and_interactions_for_ref_canary_users__(2)
        return


    def __update_sessions_and_interactions_for_canary_user__(self, 
                                                             user_id, 
                                                             ref_session_lengths:dict,
                                                             ref_session_interactions:dict,
                                                             canary_category):
        
        # Delete existing sessions and interactions for this canary user
        #self.dbManager.delete_all_session_data_for_user(user_id)
        
        # Start Updating sessions and interactions
        for session_id in ref_session_lengths.keys():
            session_length = ref_session_lengths[session_id]
            
            if(canary_category == 2):
                # For canary category 2, randomly decrease session length by 1
                if session_length > 1:
                    session_length = session_length - random.randint(0,1)

            insert_session_query = "INSERT INTO sessions (user_id, session_depth) VALUES (?, ?);"
            new_session_id = self.dbManager.execute_query(insert_session_query, (user_id, session_length))
           
            # Now insert into session_details            
            seq_num = 0
            for tool_id in ref_session_interactions[session_id]:    
                insert_session_interaction_query = "INSERT INTO session_interactions (session_id, tool_id, sequence_number) VALUES (?, ?, ?);"
                self.dbManager.execute_query(insert_session_interaction_query, (new_session_id, tool_id, seq_num))
                seq_num += 1
                if(seq_num >= session_length): 
                    break
        return
                
    def __create_sessions_and_interactions_for_all_users__(self):
        print("Creating sessions and interactions for all users...")
        self.__create_canary_users_tables__() # This should be done first
        for user_id in tqdm(range(1, CConfig.MAX_USERS + 1)):
            self.__create_sessions_and_interactions_for_user__(user_id)

    def __assign_sessions_and_interactions_for_ref_canary_users__(self, canary_category):
        if canary_category == 1:
            print(f"Get sessions and interactions for canary #1 ref. user id {self.ref_canary_1_user_id}  ...")
            # Get session length based on user id
            self.ref_canary_1_session_lengths = self.dbManager.get_all_session_lengths(self.ref_canary_1_user_id)
            self.ref_canary_1_session_interactions = self.dbManager.get_session_interactions(self.ref_canary_1_user_id)

        if canary_category == 2:
            print(f"Get sessions and interactions for canary #2 ref. user id {self.ref_canary_2_user_id}  ...")
            # Get session length based on user id
            self.ref_canary_2_session_lengths = self.dbManager.get_all_session_lengths(self.ref_canary_2_user_id)
            self.ref_canary_2_session_interactions = self.dbManager.get_session_interactions(self.ref_canary_2_user_id)

    """
    def __update_sessions_and_interactions_for_all_canary_users__(self):
        self.dbManager.delete_data_from_tables(['canary_users'])
        self.__create_canary_users__()

        # Get all Canary Users from DB
        allCanaryUsers = self.dbManager.get_canary_users()

        # Pick 1 canary user as reference
        ref_canary_1_user_idx = random.sample(range(0, len(allCanaryUsers[1])),1)[0] # This returns a list so take 1
        ref_canary_2_user_idx = random.sample(range(0, len(allCanaryUsers[2])),1)[0] # This returns a list so take 1

        ref_canary_1_user_id = allCanaryUsers[1][ref_canary_1_user_idx]
        ref_canary_2_user_id = allCanaryUsers[2][ref_canary_2_user_idx]

        # Randomly pick a reference user for each canary user
        # Create a new list of all users which are not canary users
        allCanaryUsers[1].remove(ref_canary_1_user_id) 
        allCanaryUsers[2].remove(ref_canary_2_user_id)
    
        print(f"Updating sessions and interactions for canary #1 using ref. user id {ref_canary_1_user_id}  ...")
        # Get session length based on user id
        ref_session_lengths = self.dbManager.get_all_session_lengths(ref_canary_1_user_id)
        ref_session_interactions = self.dbManager.get_session_interactions(ref_canary_1_user_id)
        
        for idx in tqdm(range(0, len(allCanaryUsers[1]))):
            self.__update_sessions_and_interactions_for_canary_user__(allCanaryUsers[1][idx],
                                                                      ref_session_lengths,
                                                                      ref_session_interactions,1) 
        
        print(f"Updating sessions and interactions for canary #2 using ref. user id {ref_canary_2_user_id}  ...")
        # Get session length based on user id
        ref_session_lengths = self.dbManager.get_all_session_lengths(ref_canary_2_user_id)
        ref_session_interactions = self.dbManager.get_session_interactions(ref_canary_2_user_id)
        
        for idx in tqdm(range(0, len(allCanaryUsers[2]))):
            self.__update_sessions_and_interactions_for_canary_user__(allCanaryUsers[2][idx],
                                                                      ref_session_lengths,
                                                                      ref_session_interactions,2) 
        return
    """

    def generate_synthetic_data(self):
        dataGenerator = CGenerateSyntheticData()
        # Order is important here
        dataGenerator.__create_users__()
        dataGenerator.__create_mcp_servers_and_tools__()
        dataGenerator.__create_sessions_and_interactions_for_all_users__()
        #dataGenerator.__update_sessions_and_interactions_for_all_canary_users__()

if __name__ == "__main__":
    CDatabaseManager.delete_db_file()
    dataGenerator = CGenerateSyntheticData() # This will create the file so dont move it earlier
    dataGenerator.generate_synthetic_data()
    #dataGenerator.__update_sessions_and_interactions_for_all_canary_users__()
    