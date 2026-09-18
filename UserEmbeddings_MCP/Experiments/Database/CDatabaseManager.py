import os,sys

# Ensure the root folder path is in sys.path ---
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------

from Experiments.CConfig import CConfig
from Experiments.Database.CSQLLite import CSQLLite

class CDatabaseManager:
    def __init__(self):
        self.sqlLite = CSQLLite(CDatabaseManager.get_database_file_path())

    @staticmethod
    def get_database_file_path() -> str:
        folderPath =    os.path.dirname(
                        os.path.dirname( #Experiments
                        #os.path.dirname( #UserEmbeddings
                        os.path.abspath(__file__)))
        dbFolder = os.path.join(folderPath, "Data")
        dbFilePath = os.path.join(dbFolder, CConfig.DB_FILE_NAME)
        return dbFilePath
    
    @staticmethod
    def delete_db_file():
        db_file_path = CDatabaseManager.get_database_file_path()
        if os.path.exists(db_file_path):
            os.remove(db_file_path)
            print(f"Database file {db_file_path} deleted.")
        else:
            print(f"Database file {db_file_path} does not exist.")

    '''
    MCPServers: id, no_of_tools
    Tools: id, mcp_id
    Users: id 
    Sessions: id, user_id, session_depth
    SessionDetails: id, session_id, tool_id,  sequence_number
    '''
    def create_tables(self):
        # Create tables if they do not exist
        create_servers_table = """
        CREATE TABLE IF NOT EXISTS mcp_servers (
            id INTEGER PRIMARY KEY,
            no_of_tools INTEGER
        );
        """

        create_tools_table = """
        CREATE TABLE IF NOT EXISTS mcp_tools (
            id INTEGER PRIMARY KEY,
            mcp_server_id INTEGER,
            mcp_tool_id INTEGER,
            FOREIGN KEY (mcp_server_id) REFERENCES mcp_servers (id)
        );
        """
        
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY
        );
        """

        create_sessions_table = """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            session_depth INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """

        create_session_interactions_table = """
            CREATE TABLE IF NOT EXISTS session_interactions (
            id INTEGER PRIMARY KEY,
            session_id INTEGER,
            tool_id INTEGER,
            sequence_number INTEGER,
            FOREIGN KEY (session_id) REFERENCES users (id)
            FOREIGN KEY (tool_id) REFERENCES mcp_tools (id)
        );
        """

        create_canary_users = """
        CREATE TABLE IF NOT EXISTS canary_users (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            canary_category INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """
        
        self.sqlLite.execute_query(create_servers_table)
        self.sqlLite.execute_query(create_tools_table)
        self.sqlLite.execute_query(create_users_table)
        self.sqlLite.execute_query(create_sessions_table)
        self.sqlLite.execute_query(create_canary_users)
        self.sqlLite.execute_query(create_session_interactions_table)

    def last_insert_rowid(self):
        (id,) = self.dbManager.execute_read_query("SELECT last_insert_rowid();")[0]
        return id

    def execute_query(self, query, params=()):
        return self.sqlLite.execute_query(query, params)
    
    def execute_read_query(self, query, params=()):
        return self.sqlLite.execute_read_query(query, params)

    #------------- Read specific data methods ----------------
    # Returns a dictionary with session_id as key and session_length as value
    def get_all_session_lengths(self, user_id):
        query = "SELECT id,session_depth FROM sessions WHERE user_id = ?;"
        result = self.execute_read_query(query, (user_id,))
        allSessionLengths = {}
        for row in result:
            session_id = row[0]
            session_length = row[1]
            allSessionLengths[session_id] = session_length
        return allSessionLengths

    # Returns a dictionaty with session_id as key and tool_id as value
    def get_session_interactions(self, user_id):
        query = """
        SELECT si.session_id, si.tool_id 
        FROM session_interactions si
        JOIN sessions s ON si.session_id = s.id
        WHERE s.user_id = ?;
        """
        allInteractions = {}
        result = self.execute_read_query(query, (user_id,))
        interactions = [(row[0], row[1]) for row in result]
        for (session_id, tool_id) in interactions:
            if session_id not in allInteractions:
                allInteractions[session_id] = []
            allInteractions[session_id].append(tool_id)
        return allInteractions


    def get_number_of_tools_for_server(self, mcp_server_id):
        query = "SELECT no_of_tools FROM mcp_servers WHERE id = ?;"
        result = self.execute_read_query(query, (mcp_server_id,))
        if result:
            return result[0][0]
        return 0
    
    def get_all_user_ids(self):
        query = "SELECT id FROM users;"
        result = self.execute_read_query(query)
        user_ids = [row[0] for row in result]
        return user_ids
    
    def get_sessions_for_user(self, user_id):
        query = "SELECT id FROM sessions WHERE user_id = ?;"
        result = self.execute_read_query(query, (user_id,))
        session_ids = [row[0] for row in result]
        return session_ids
    
    def get_tools_for_session(self, session_id):
        query = "SELECT tool_id FROM session_interactions WHERE session_id = ?;"
        result = self.execute_read_query(query, (session_id,))
        tool_ids = [row[0] for row in result]
        return tool_ids
    
    def get_all_tools_and_sessions(self):
        query = "SELECT session_id, tool_id FROM session_interactions;"
        result = self.execute_read_query(query, ())
        # return as dictionary with session_id as key and list of tool_ids as value    
        all_sessions_data = {}
        for row in result: # couple of million+ rows
            session_id = row[0]
            tool_id = row[1]
            if session_id not in all_sessions_data:
                all_sessions_data[session_id] = []
            all_sessions_data[session_id].append(tool_id)
        return all_sessions_data
    
    def get_number_of_tools(self):
        query = "SELECT COUNT(*) FROM mcp_tools;"
        result = self.execute_read_query(query)
        if result:
            return result[0][0]
        return 0
    
    def get_all_tool_ids(self):
        query = "SELECT id FROM mcp_tools;"
        result = self.execute_read_query(query)
        tool_ids = [row[0] for row in result]
        return tool_ids
    
    def get_tool_call_count(self, tool_id):
        query = "SELECT COUNT(*) FROM session_interactions WHERE tool_id = ?;"
        result = self.execute_read_query(query, (tool_id,))
        if result:
            return result[0][0]
        return 0
    
    def get_all_mcp_server_ids(self):
        query = "SELECT id FROM mcp_servers;"
        result = self.execute_read_query(query)
        server_ids = [row[0] for row in result]
        return server_ids
    
    def get_canary_users(self):
        query = "SELECT user_id, canary_category FROM canary_users;"
        result = self.execute_read_query(query)
        canary_users = {}
        for row in result:
            user_id = row[0]
            canary_category = row[1]
            if canary_category not in canary_users:
                canary_users[canary_category] = []
            canary_users[canary_category].append(user_id)
        return canary_users

    #------------- END: Read specific data methods ----------------
    #------------- Delete specific data methods ----------------
    def delete_all_session_data_for_user(self, user_id):
        # First get all session ids for the user
        get_sessions_query = "SELECT id FROM sessions WHERE user_id = ?;"
        session_ids = self.execute_read_query(get_sessions_query, (user_id,))
        session_ids = [row[0] for row in session_ids]
        
        # Delete session interactions for these sessions
        delete_interactions_query = "DELETE FROM session_interactions WHERE session_id = ?;"
        for session_id in session_ids:
            self.execute_query(delete_interactions_query, (session_id,))
        
        # Delete sessions for the user
        delete_sessions_query = "DELETE FROM sessions WHERE user_id = ?;"
        self.execute_query(delete_sessions_query, (user_id,))
    
    #------------- END: Delete specific data methods ----------------

    def delete_data_from_tables(self, table_names):
        for table_name in table_names:
            delete_query = f"DELETE FROM {table_name};"
            self.sqlLite.execute_query(delete_query)
    
    def delete_all_data(self):
        self.delete_data_from_tables([
            'session_interactions',
            'sessions',
            'users',
            'mcp_tools',
            'mcp_servers',
            'canary_users'
        ])


if __name__ == "__main__": # For testing purposes
    #CDatabaseManager.delete_db_file()
    dbManager = CDatabaseManager()
    all_session_data =  dbManager.get_all_tools_and_sessions()
    print(f"Total sessions: {len(all_session_data)}")
    #dbManager.create_tables()