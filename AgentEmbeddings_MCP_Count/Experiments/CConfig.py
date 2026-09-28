from dataclasses import dataclass

# For changes in embedding dimension, make sure to update the file names 
# and size accordingly
@dataclass(frozen=True)
class CConfig:
    NO_TOOL_CALL_ID = -1
    MAX_AGENTS = 50000 # Change the file names accordingly
    DB_FILE_NAME = "mcp_interactions_a50000.db"
    BASE_EMBEDDINGS_FILE_NAME = "agent_embeddings_a50000.pt"
    BASE_TRAINING_LOSS_FILE_NAME = "training_loss_a50000.pkl" # Pickle file
    MAX_MCP_SERVERS = 200
    MAX_TOOLS_PER_MCP_SERVER = 30
    MIN_TOOLS_PER_MCP_SERVER = 1
    SESSIONS_PER_AGENT_MEAN = 300
    SESSIONS_PER_AGENT_STD = 200
    MAX_SESSIONS_PER_AGENT = 1000
    SESSIONS_LENGTH_MEAN = 30
    SESSIONS_LENGTH_STD = 20
    MIN_TOOL_CALLS_PER_SEQUENCE = 0
    MAX_TOOL_CALLS_PER_SEQUENCE = 5
    EMBEDDING_DIMENSIONS = 8 # Embedding vector dimensions
    EMBEDDING_DIMENSIONS_TO_RUN = (8, 24)
    MAX_PAIR_SAMPLES = 1000000
    SILHOUETTE_SAMPLE_SIZE = 5000
    PROB_OF_TOOL_FROM_SAME_MCP = 0.33
    PERCENTAGE_AGENTS_CANARY_1 = 5
    PERCENTAGE_AGENTS_CANARY_2 = 5
    EMBEDDINGS_FILE_NAME = "agent_embeddings.pt"
