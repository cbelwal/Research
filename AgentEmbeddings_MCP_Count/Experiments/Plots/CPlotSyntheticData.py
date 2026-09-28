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
    def __init__(self, outputDirectory=None):
        self.dbManager = CDatabaseManager()
        if outputDirectory is not None:
            CPlotCommon.set_output_directory(outputDirectory)
    
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

        
    def plot_number_of_sessions_per_agent(self):
        all_agent_sessions = [
            count
            for _, count in self.dbManager.get_session_counts_per_agent()
        ]
        
        # There will be lot of agents so plot with histogram
        CPlotCommon.plot_histogram_y(all_agent_sessions,
                                    bins=30,                                     
                                    title="Distribution: Number of Sessions / Agent",
                                    xlabel="Number of Sessions",
                                    ylabel="Agent Count",
                                    saveFile=True)

    
    def plot_calls_for_each_tool(self):
        all_tool_call_count = [
            count
            for _, count in self.dbManager.get_tool_call_counts()
            if count > 0
        ]
        # There will be lot of agents so plot with histogram
        CPlotCommon.plot_histogram_y(all_tool_call_count,
                                    bins=30,                                     
                                    title="Distribution: Number of Calls / Tool",
                                    xlabel="Number of Tool Calls",
                                    ylabel="Tool Count",
                                    saveFile=True)
    
    def generate_all_plots(self):
        self.plot_tools_in_each_mcp()
        self.plot_number_of_sessions_per_agent()
        self.plot_calls_for_each_tool()

if __name__== "__main__":
    plotter = CPlotSyntheticData()
    plotter.plot_tools_in_each_mcp()