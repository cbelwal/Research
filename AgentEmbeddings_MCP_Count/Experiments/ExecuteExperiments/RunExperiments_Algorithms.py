import os,sys
import argparse
import datetime
import hashlib
import json
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------

from Algorithms.Alg_1_DataPreparation import BuildAgentToolMatrix
from Algorithms.Alg_2_AutoEncoder import Alg_2_AutoEncoder
from Algorithms.Alg_3_PolynomialFit import Alg_3_PolynomialFit
from Algorithms.Alg_Baseline_PCA import Alg_Baseline_PCA

from Algorithms.Helpers.CDataMain import CDataMain
from Experiments.CConfig import CConfig
from Experiments.ExecuteExperiments.Helpers.CResultsStore import CResultsStore
from Experiments.ExecuteExperiments.Helpers.CDistanceAnalysis import CDistanceAnalysis


def run_algorithms_on_synthetic_data(agent_tool_matrix, algID:int, embeddingDimensions:int):
    # Step 2: Create Test Data Main Instance
    testData = CDataMain(agent_tool_matrix)
    if(algID == 2):
        (MAT_E, loss_for_each_agent) = Alg_2_AutoEncoder(
                embeddingDimensions=embeddingDimensions,
                testData=testData)
    elif(algID == 3):
        (MAT_E, loss_for_each_agent) = Alg_3_PolynomialFit(
                embeddingDimensions=embeddingDimensions,
                testData=testData)
    else:
        raise ValueError(f"Unsupported algorithm ID: {algID}")
    print(f"Generated Embeddings with Algorithm {algID}") 
    return (MAT_E, loss_for_each_agent)

def run_pca_on_synthetic_data(All_C_hat_a:dict,algID:int):
    testData = CDataMain(All_C_hat_a)
    embeddingDimensions = CConfig.EMBEDDING_DIMENSIONS
   
    (MAT_E, loss_for_each_agent) = Alg_Baseline_PCA(
                embeddingDimensions=embeddingDimensions,
                testData=testData)
    print(f"Generated Embeddings with PCA") 
    return (MAT_E, loss_for_each_agent)


def store_results_in_file(
    MAT_E,
    loss_for_each_agent,
    algID:int,
    embeddingDimensions:int,
    metadata:dict,
):
    store = CResultsStore(
        algID=algID,
        embeddingDimensions=embeddingDimensions,
    )
    # Store embeddings in file
    store.store_embeddings(MAT_E)
    print(f"Agent embeddings stored in file")
    # Store training loss in file
    store.store_training_loss(loss_for_each_agent)
    store.store_metadata(metadata)
    print(f"Training loss stored in file")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_data_fingerprint():
    database_path = os.path.join(
        topRootPath,
        "Experiments",
        "Data",
        CConfig.DB_FILE_NAME,
    )
    def sqlite_file_identity(path):
        if not os.path.isfile(path):
            return None
        file_stat = os.stat(path)
        return {
            "size": file_stat.st_size,
            "modified_ns": file_stat.st_mtime_ns,
        }

    payload = {
        "database_path": os.path.abspath(database_path),
        "sqlite_files": {
            "database": sqlite_file_identity(database_path),
            "wal": sqlite_file_identity(f"{database_path}-wal"),
            "shared_memory": sqlite_file_identity(f"{database_path}-shm"),
        },
        "no_tool_call_id": CConfig.NO_TOOL_CALL_ID,
        "max_agents": CConfig.MAX_AGENTS,
        "max_mcp_servers": CConfig.MAX_MCP_SERVERS,
        "min_tools_per_server": CConfig.MIN_TOOLS_PER_MCP_SERVER,
        "max_tools_per_server": CConfig.MAX_TOOLS_PER_MCP_SERVER,
        "sessions_per_agent_mean": CConfig.SESSIONS_PER_AGENT_MEAN,
        "sessions_per_agent_std": CConfig.SESSIONS_PER_AGENT_STD,
        "max_sessions_per_agent": CConfig.MAX_SESSIONS_PER_AGENT,
        "session_length_mean": CConfig.SESSIONS_LENGTH_MEAN,
        "session_length_std": CConfig.SESSIONS_LENGTH_STD,
        "min_tool_calls_per_sequence": CConfig.MIN_TOOL_CALLS_PER_SEQUENCE,
        "max_tool_calls_per_sequence": CConfig.MAX_TOOL_CALLS_PER_SEQUENCE,
        "same_mcp_probability": CConfig.PROB_OF_TOOL_FROM_SAME_MCP,
        "matrix_source_sha256": {
            path: file_sha256(os.path.join(topRootPath, path))
            for path in (
                "Algorithms/Alg_1_DataPreparation.py",
                "Experiments/Database/CDataPreparationHelper.py",
                "Experiments/Database/CDatabaseManager.py",
                "Experiments/Database/CSQLLite.py",
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_experiment_metadata(
    data_fingerprint,
    algorithm_id,
    embedding_dimensions,
):
    source_paths = {
        2: [
            os.path.join(topRootPath, "Algorithms", "Alg_2_AutoEncoder.py"),
            os.path.join(
                topRootPath,
                "Algorithms",
                "Helpers",
                "CAgentToolAutoencoder.py",
            ),
        ],
        3: [
            os.path.join(topRootPath, "Algorithms", "Alg_3_PolynomialFit.py"),
            os.path.join(
                topRootPath,
                "Algorithms",
                "Helpers",
                "CPolynomialFitReduction.py",
            ),
        ],
    }
    return {
        "data_fingerprint": data_fingerprint,
        "algorithm_id": algorithm_id,
        "embedding_dimensions": embedding_dimensions,
        "source_sha256": {
            os.path.relpath(path, topRootPath): file_sha256(path)
            for path in source_paths[algorithm_id]
        },
    }


# Run the experiments and store embeddings and losses in file
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(CConfig.EMBEDDING_DIMENSIONS_TO_RUN),
    )
    parser.add_argument(
        "--algorithms",
        type=int,
        nargs="+",
        default=[2, 3],
        choices=[2, 3],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the matrix cache and recompute existing results.",
    )
    return parser.parse_args()


if __name__== "__main__":
    args = parse_args()
    print(f"Starting experiments on synthetic data at time {datetime.datetime.now()}...")
    print("Running A/g #1 on synthetic data...")
    results_root = os.path.join(
        topRootPath,
        "Experiments",
        "Data",
        "ExperimentResults",
        f"a{CConfig.MAX_AGENTS}",
    )
    matrix_cache = os.path.join(results_root, "agent_tool_matrix.pt")
    data_fingerprint = build_data_fingerprint()
    agent_tool_matrix = BuildAgentToolMatrix(
        cache_path=matrix_cache,
        cache_fingerprint=data_fingerprint,
        force=args.force,
    )

    for embedding_dimensions in args.dimensions:
        for algID in args.algorithms:
            result_store = CResultsStore(
                algID=algID,
                embeddingDimensions=embedding_dimensions,
            )
            embeddings_path = result_store.get_file_path(
                CConfig.BASE_EMBEDDINGS_FILE_NAME
            )
            loss_path = result_store.get_file_path(
                CConfig.BASE_TRAINING_LOSS_FILE_NAME
            )
            expected_metadata = build_experiment_metadata(
                data_fingerprint,
                algID,
                embedding_dimensions,
            )
            if (
                not args.force
                and os.path.isfile(embeddings_path)
                and os.path.isfile(loss_path)
                and result_store.load_metadata() == expected_metadata
            ):
                print(
                    f"Skipping Algorithm {algID}, dimension "
                    f"{embedding_dimensions}; results already exist."
                )
                continue
            print(
                f"Computing embeddings for Algorithm {algID} "
                f"with dimension {embedding_dimensions}..."
            )
            (MAT_E, loss_for_each_agent) = run_algorithms_on_synthetic_data(
                agent_tool_matrix,
                algID,
                embedding_dimensions,
            )
            print(f"Storing embeddings for Algorithm {algID}...")
            store_results_in_file(
                MAT_E,
                loss_for_each_agent,
                algID=algID,
                embeddingDimensions=embedding_dimensions,
                metadata=expected_metadata,
            )
    print(f"Experiments completed at time {datetime.datetime.now()}.")

       
 