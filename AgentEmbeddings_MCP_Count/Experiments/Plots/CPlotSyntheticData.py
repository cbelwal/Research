"""
Routines to plot orig. data directly from the database
"""
import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.Database.CDatabaseManager import CDatabaseManager
from Experiments.Plots.CPlotCommon import CPlotCommon

class CPlotSyntheticData:
    def __init__(self):
        self.dbManager = CDatabaseManager()
        
    
    def plot_tools_in_each_mcp(self):
        all_tool_count = []
        mcp_server_ids = self.dbManager.get_all_mcp_server_ids()
        for mcp_id in mcp_server_ids:
            tool_count = self.dbManager.get_number_of_tools_for_server(mcp_id)
            all_tool_count.append(tool_count)
        
        CPlotCommon.plot_bar_y(all_tool_count,                                     
                                     title="Number of Tools in each MCP Server",
                                     xlabel="MCP Server Id",
                                     ylabel="Number of Tools",
                                     saveFile=True)
        return

        
    def plot_number_of_sessions_per_user(self):
        all_user_sessions = []
        user_ids = self.dbManager.get_all_user_ids()
        for user_id in user_ids:
            userSession = self.dbManager.get_sessions_for_user(user_id)
            all_user_sessions.append(len(userSession))
        
        # There will be lot of users so plot with histogram
        CPlotCommon.plot_histogram_y(all_user_sessions,
                                    bins=30,                                     
                                    title="Distribution: Number of Sessions / User",
                                    xlabel="Number of Sessions",
                                    ylabel="User Count",
                                    saveFile=True)

    
    def plot_calls_for_each_tool(self):
        all_tool_call_count = []
        tools_ids = self.dbManager.get_all_tool_ids()
        for tool_id in tools_ids:
            tool_call_count = self.dbManager.get_tool_call_count(tool_id)
            if tool_call_count > 0:
                all_tool_call_count.append(tool_call_count)
        # There will be lot of users so plot with histogram
        CPlotCommon.plot_histogram_y(all_tool_call_count,
                                    bins=30,                                     
                                    title="Distribution: Number of Calls / Tool",
                                    xlabel="Number of Tool Calls",
                                    ylabel="Tool Count",
                                    saveFile=True)
    
    def generate_all_plots(self):
        self.plot_tools_in_each_mcp()
        self.plot_number_of_sessions_per_user()
        self.plot_calls_for_each_tool()

if __name__== "__main__":
    plotter = CPlotSyntheticData()
    plotter.plot_tools_in_each_mcp()