from dataclasses import dataclass

# For changes in embedding dimension, make sure to update the file names 
# and size accordingly
@dataclass(frozen=True)
class CConfig:
    MAX_USERS = 10000 # Change the file names accordingly
    DB_FILE_NAME = "mcp_interactions_u10000.db"
    BASE_EMBEDDINGS_FILE_NAME = "user_embeddings_u10000.pt"
    BASE_TRAINING_LOSS_FILE_NAME = "training_loss_u10000.pkl" # Pickle file
    MAX_MCP_SERVERS = 100
    MAX_TOOLS_PER_MCP_SERVER = 50
    MIN_TOOLS_PER_MCP_SERVER = 1
    SESSIONS_PER_USER_MEAN = 100
    SESSIONS_PER_USER_STD = 60 
    SESSIONS_LENGTH_MEAN = 20
    SESSIONS_LENGTH_STD = 10
    EMBEDDING_DIMENSIONS = 8 # Embedding vector dimensions
    PROB_OF_TOOL_FROM_SAME_MCP = 0.33
    PERCENTAGE_USERS_CANARY_1 = 5
    PERCENTAGE_USERS_CANARY_2 = 5
    EMBEDDINGS_FILE_NAME = "user_embeddings.pt"
