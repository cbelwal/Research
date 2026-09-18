class CCache:
    def __init__(self):
        self.mcp_tools = {}
        self.user_ids = set()
        pass

    def add_user(self, user_id:int):
        self.user_ids.add(user_id)
    
    def get_all_user_ids(self):
        return list(self.user_ids)

    def add_mcp_tool(self, mcp_server_id:int, mcp_tool_id:int, tool_id:int):
        if mcp_server_id not in self.mcp_tools:
            self.mcp_tools[mcp_server_id] = {}
        self.mcp_tools[mcp_server_id][mcp_tool_id] = tool_id
    
    def get_tool_id(self, mcp_server_id:int, mcp_tool_id:int):
        if mcp_server_id in self.mcp_tools:
            return self.mcp_tools[mcp_server_id][mcp_tool_id]
        return None
    
    def get_number_of_tools_for_server(self, mcp_server_id:int):
        if mcp_server_id in self.mcp_tools:
            return len(self.mcp_tools[mcp_server_id])
        return 0
        