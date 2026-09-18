# UserEmbeddings_MCP

A machine learning research system that generates low-dimensional **user embeddings** from user interaction data with **MCP (Model Context Protocol)** tools. The system models user behavior patterns through their tool usage and creates compact vector representations (embeddings) that enable user similarity analysis, clustering, and recommendation tasks.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Setup: Virtual Environment](#3-setup-virtual-environment)
4. [Project Structure](#4-project-structure)
5. [Configuration](#5-configuration)
6. [Data: Synthetic Generation & Database](#6-data-synthetic-generation--database)
7. [Algorithm 1: Data Preparation](#7-algorithm-1-data-preparation)
8. [Algorithm 2: Shared Autoencoder Embeddings](#8-algorithm-2-shared-autoencoder-embeddings)
9. [Algorithm 3: Matrix Factorization](#9-algorithm-3-matrix-factorization)
10. [Baseline: PCA Embeddings](#10-baseline-pca-embeddings)
11. [Analysis & Evaluation](#11-analysis--evaluation)
12. [Visualization & Plots](#12-visualization--plots)
13. [Running the Code](#13-running-the-code)
14. [End-to-End Data Flow](#14-end-to-end-data-flow)
15. [Important File Reference](#15-important-file-reference)
16. [Key Design Decisions](#16-key-design-decisions)

---

## 1. Project Goal

The goal of this project is to answer the question:

> **Can we represent a user's behavior — specifically, which MCP tools they use and how often — as a short fixed-length vector (an "embedding"), and do similar users end up with similar embeddings?**

### What is an MCP Tool?

MCP (Model Context Protocol) servers expose *tools* — callable functions that an AI assistant can invoke. A single MCP server might provide 1–50 tools (e.g., file reading, web search, database queries). Users interact with these tools across many sessions.

### What is a User Embedding?

An embedding is a short, dense numeric vector (e.g., 8 numbers) that summarizes a user's behavioral profile. Users who tend to use the same tools in similar proportions should end up close together in embedding space.

### Why Does This Matter?

- **User Similarity:** Find users with similar tool usage habits.
- **Clustering:** Group users with similar behavior automatically.
- **Recommendations:** Suggest tools to users based on similar users' behavior.
- **Anomaly Detection:** Identify users whose behavior deviates significantly from peers.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Synthetic Data Generation                  │
│   (Users, MCP Servers, Tools, Sessions, Interactions)   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   SQLite Database                       │
│   mcp_interactions_u10000.db                            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│          Algorithm 1: Data Preparation                  │
│   Normalize tool usage frequencies → sparse matrix      │
│   Output: {user_id: {tool_id: frequency}} dict          │
└──────────┬──────────────┬──────────────┬────────────────┘
           │              │              │
           ▼              ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Algorithm 2  │  │ Algorithm 3  │  │  PCA         │
│ Shared       │  │ Matrix       │  │  Baseline    │
│ Autoencoder  │  │ Factorization│  │              │
│              │  │              │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────┬────────┘                 │
                ▼                          ▼
┌───────────────────────┐    ┌─────────────────────────┐
│  Embeddings stored    │    │  Baseline embeddings    │
│  user_embeddings_     │    │  stored separately      │
│  alg_N.pt             │    │                         │
└───────────┬───────────┘    └─────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│                  Analysis & Evaluation                  │
│  - Canary user distance validation                      │
│  - Top-K similarity search                             │
│  - K-Means clustering (elbow method)                    │
│  - Cosine & Euclidean distance histograms               │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                     Visualizations                      │
│  - PCA 2D projections, cluster plots, distance plots    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Setup: Virtual Environment

> **Important:** Always use a virtual environment to avoid dependency conflicts.

### Step-by-Step Setup

```bash
# 1. Create the virtual environment in the project root
py -m venv .venv

# 2. Activate it
. .\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # macOS / Linux

# 3. Verify you are using the venv Python
where python                 # Windows
# which python               # macOS / Linux
# Should print a path inside .venv\

# 4. Upgrade pip
py -m pip install --upgrade pip

# 5. Install all dependencies
pip install -r requirements.txt
```

### Dependencies (requirements.txt)

| Package | Purpose |
|---------|---------|
| `torch` | Neural network framework used in Algorithm 2 |
| `numpy` | Numerical computation, polynomial fitting |
| `pandas` | Data manipulation and analysis |
| `scikit-learn` | PCA, K-Means clustering, StandardScaler |
| `tqdm` | Progress bars during long loops |
| `matplotlib` | All plotting and visualization |
| `kneed` | Automatic elbow point detection in clustering curves |

---

## 4. Project Structure

```
UserEmbeddings_MCP/
│
├── Algorithms/                          # Core algorithm implementations
│   ├── README.md                        # Algorithm contracts and usage
│   ├── Alg_1_DataPreparation.py         # Step 1: Normalize raw interaction data
│   ├── Alg_2_AutoEncoder.py             # Step 2a: Shared autoencoder embeddings
│   ├── Alg_3_MatrixFactorization.py     # Step 2b: Truncated-SVD embeddings
│   ├── Alg_Baseline_PCA.py              # PCA baseline
│   ├── Alg_Data_Raw.py                  # Raw (unnormalized) data extraction
│   ├── Unused/                           # Retired algorithm implementations
│   └── Helpers/
│       ├── IUserToolMatrix.py           # Abstract interface for data matrices
│       ├── CDataMain.py                 # Converts dict → PyTorch tensor matrix
│       ├── CUserToolAutoencoder.py      # Shared encoder/decoder model
│       ├── CSingleLayer.py              # Legacy single-layer NN module
│       ├── CModelTraining.py            # Training loop (SGD + MSE loss)
│       └── CPolynomialFitReduction.py   # Polynomial coefficient extraction
│
├── Experiments/
│   ├── CConfig.py                       # Central configuration constants
│   │
│   ├── DataGeneration/
│   │   ├── CGenerateSyntheticData.py    # Generates synthetic DB from scratch
│   │   ├── CCache.py                    # In-memory cache to speed up DB writes
│   │   ├── GenerateSyntheticData.py     # Entry point to regenerate the database
│   │   └── PlotSyntheticData.py         # Visualize raw synthetic data properties
│   │
│   ├── Database/
│   │   ├── CDatabaseManager.py          # High-level DB query interface
│   │   ├── CSQLLite.py                  # Low-level SQLite wrapper with tuning
│   │   ├── CDataPreparationHelper.py    # Cached DB reads for Algorithm 1
│   │   └── __init__.py
│   │
│   ├── ExecuteExperiments/
│   │   ├── RunExperiments_Algorithms.py # MAIN ENTRY POINT: run algs 2 and 3
│   │   ├── RunExperiments_Baseline_PCA.py  # Entry point: run PCA baseline
│   │   ├── PlotExperimentalData.py      # MAIN ENTRY POINT: generate all plots
│   │   ├── PlotBaselineClustering.py    # Entry point: plot PCA baseline results
│   │   └── Helpers/
│   │       ├── CResultsStore.py         # Save/load embeddings (.pt) & losses (.pkl)
│   │       ├── CDistanceAnalysis.py     # Orchestrates distance computations
│   │       ├── CDistanceFunctions.py    # Cosine & Euclidean distance formulas
│   │       ├── CClusteringAnalysis.py   # K-Means + elbow point detection
│   │       ├── CTopKUsers.py            # Top-K most similar users lookup
│   │       └── CPCAAnalysis.py          # 2D PCA projection for visualization
│   │
│   ├── Plots/
│   │   ├── CPlotExperimentalData.py     # Orchestrates which plots to generate
│   │   ├── CPlotCommon.py               # Shared matplotlib utilities
│   │   ├── CPlotDistance.py             # Distance histogram plots
│   │   └── CPlotSyntheticData.py        # Raw data distribution plots
│   │
│   └── Data/
│       ├── mcp_interactions_u10000.db   # Pre-generated SQLite database (10,000 users)
│       └── ExperimentResults/           # Output: .pt embeddings + .pkl loss files
│
├── requirements.txt
└── README.md
```

See [`Algorithms/README.md`](Algorithms/README.md) for the active algorithm
contracts, output formats, tuning parameters, and test commands.

---

## 5. Configuration

All global parameters live in **`Experiments/CConfig.py`**. This is the single file to change when scaling experiments.

```python
class CConfig:
    MAX_USERS = 10000              # Total users in the synthetic dataset
    DB_FILE_NAME = "mcp_interactions_u10000.db"  # Must match MAX_USERS

    MAX_MCP_SERVERS = 100          # Number of MCP servers in the simulation
    MAX_TOOLS_PER_MCP_SERVER = 50  # Max tools per server (min = 1)
    MIN_TOOLS_PER_MCP_SERVER = 1

    SESSIONS_PER_USER_MEAN = 100   # Average sessions per user (normal dist)
    SESSIONS_PER_USER_STD  = 60
    SESSIONS_LENGTH_MEAN = 20      # Average tools called per session
    SESSIONS_LENGTH_STD  = 10

    EMBEDDING_DIMENSIONS = 8       # Size of the output embedding vector

    PROB_OF_TOOL_FROM_SAME_MCP = 0.33  # Probability of tool clustering behavior

    PERCENTAGE_USERS_CANARY_1 = 5  # % of users who are exact canary duplicates
    PERCENTAGE_USERS_CANARY_2 = 5  # % of users who are slight canary variants
```

> **Note:** If you change `MAX_USERS`, you must also update `DB_FILE_NAME` to match (and regenerate the database). Similarly, if you change `EMBEDDING_DIMENSIONS`, re-run the embedding algorithms.

---

## 6. Data: Synthetic Generation & Database

### Why Synthetic Data?

Real MCP interaction logs may not be publicly available. Synthetic data allows controlled experiments where we know *exactly* which users should cluster together (via "canary users"), enabling objective evaluation.

### Database Schema

The SQLite database (`mcp_interactions_u10000.db`) has 6 tables:

```
mcp_servers          mcp_tools              users
──────────────       ─────────────────      ──────
id (PK)              id (PK)                id (PK)
no_of_tools          mcp_server_id (FK)
                     mcp_tool_id

sessions             session_interactions   canary_users
──────────────       ─────────────────      ──────────────
id (PK)              id (PK)                id (PK)
user_id (FK)         session_id (FK)        user_id (FK)
session_depth        tool_id (FK)           canary_category
                     sequence_number
```

### How Synthetic Data is Generated

**File:** `Experiments/DataGeneration/CGenerateSyntheticData.py`

1. **MCP Servers & Tools:** Creates 100 MCP servers, each with 1–50 tools randomly.
2. **Users:** Creates 10,000 users.
3. **Sessions:** For each user, draws the number of sessions from `Normal(mean=100, std=60)`.
4. **Session Interactions:** For each session:
   - Draw session length from `Normal(mean=20, std=10)`
   - For each tool call in the session:
     - With probability 0.33 → pick a tool from the *same* MCP server as the previous tool (mimicking realistic tool clustering)
     - With probability 0.67 → pick any tool at random
5. **Canary Users:** A key validation mechanism:
   - **Canary Category 1** (5% of users): Exact copies of a reference user. These users have identical session data. Any good embedding algorithm must produce nearly identical embeddings for them.
   - **Canary Category 2** (5% of users): Near-copies of the reference user with session length reduced by 1. These should produce *similar but not identical* embeddings.

### Regenerating the Database

```bash
# From the project root (with .venv activated)
python Experiments/DataGeneration/GenerateSyntheticData.py
```

> **Warning:** The existing `mcp_interactions_u10000.db` is pre-generated and ready to use. Only re-run this if you change configuration parameters.

---

## 7. Algorithm 1: Data Preparation

**File:** `Algorithms/Alg_1_DataPreparation.py`

This is always the **first step**. It converts raw interaction counts from the database into a normalized sparse matrix suitable for embedding generation.

### What It Produces

A Python dictionary of dictionaries representing a **sparse user-tool matrix**:

```
all_C_hat_u_1 = {
    user_id_1: { tool_id_A: 0.45, tool_id_B: 0.30, tool_id_C: 0.25 },
    user_id_2: { tool_id_A: 0.10, tool_id_D: 0.90 },
    ...
}
```

Each value is a normalized frequency between 0 and 1, summing to 1.0 for each user.

### Normalization Steps (per user)

**Step 1 — Count raw tool calls:**
```
C_u[tool_id] = number of times user u called tool_id across all sessions
```

**Step 2 — Normalize by total tool calls:**
```
C_hat_u[tool_id] = C_u[tool_id] / total_tool_calls_by_user_u
```

**Step 3 — Re-normalize to [0, 1] summing to 1:**
```
C_hat_u_1[tool_id] = C_hat_u[tool_id] / sum(C_hat_u.values())
```

> The double normalization ensures that the values are always in [0, 1] and sum to exactly 1, making them compatible with sigmoid activations used in Algorithm 2.

### Why Sparse?

Most users only interact with a small fraction of the ~5,000 total tools available (100 servers × up to 50 tools each). Storing the full dense matrix would be wasteful; the sparse dictionary stores only non-zero entries.

---

## 8. Algorithm 2: Shared Autoencoder Embeddings

**File:** `Algorithms/Alg_2_AutoEncoder.py`

### Concept

A single autoencoder is trained across all users. Its encoder compresses each complete user-tool usage vector into a fixed-size embedding, and its decoder reconstructs the original vector.

### Architecture

```
User-tool vector → Linear/ReLU → embedding → Linear/ReLU/Linear/Sigmoid → reconstructed vector
```

- **Input:** One normalized vector containing every tool's usage frequency
- **Encoder output:** The user embedding, shared in one coordinate system across users
- **Decoder output:** A reconstruction with the same number of columns as the input
- **Loss:** Weighted mean squared error, with observed tools weighted more heavily

### Training

**File:** `Algorithms/Helpers/CUserToolAutoencoder.py`

```
Initialize one shared encoder and decoder
Repeat up to 5,000 epochs:
    Read batches of complete user-tool vectors
    Encode each vector into embedding_dim values
    Decode each embedding back into a user-tool vector
    Update the shared model with weighted reconstruction loss
Run the trained encoder over every user to produce MAT_E
```

### Key Files

| File | Role |
|------|------|
| `Algorithms/Helpers/CUserToolAutoencoder.py` | Shared encoder and decoder network |
| `Algorithms/Alg_2_AutoEncoder.py` | Batched training and embedding extraction |
| `Algorithms/Helpers/CDataMain.py` | Converts `all_C_hat_u_1` dict → PyTorch tensor matrix |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| MAX_EPOCHS | 5,000 |
| MIN_TARGET_LOSS | 5e-5 |
| LEARNING_RATE | 0.02 |
| OPTIMIZER | Adam |
| LOSS FUNCTION | Weighted MSE |

### Output

- `MAT_E` — tensor of shape `(num_users, embedding_dim)` — the embedding matrix
- `loss_for_each_user` — list of final MSE loss per user (used to evaluate convergence)

---

## 9. Algorithm 3: Matrix Factorization

**File:** `Algorithms/Alg_3_MatrixFactorization.py`

Algorithm 3 applies truncated singular value decomposition to the complete
user-tool matrix:

`user_tool_matrix ≈ user_embeddings × tool_factors`

`TruncatedSVD` learns both factors globally. Its transformed user factors are
the embeddings, and each user's mean squared reconstruction error is returned
as that user's loss. The method is deterministic, non-neural, and independent
of tool ordering. The embedding dimension cannot exceed the smaller matrix
dimension.

---

## 10. Baseline: PCA Embeddings

**File:** `Algorithms/Alg_Baseline_PCA.py`

### Concept

Principal Component Analysis (PCA) is a classical dimensionality reduction method. Instead of a per-user model, it learns a *global* linear projection from the full user-tool matrix.

### How It Works

1. Build the full dense user-tool matrix (shape: `num_users × num_tools`)
2. Apply `StandardScaler` to normalize each tool's usage across users
3. Fit PCA with `n_components = embedding_dim` (e.g., 8)
4. Transform all users' rows through the learned projection
5. Result: each user has an 8-dimensional embedding

### Key Difference from Algorithms 2 and 3

| Aspect | Algorithms 2 and 3 | PCA Baseline |
|--------|-----------|-------------|
| Training | Shared across users | Global |
| Method | Neural / analytical | Linear projection |
| New users | Alg 2 encodes directly; Alg 3 requires refitting | Can project with the fitted PCA model |

**Entry point:** `Experiments/ExecuteExperiments/RunExperiments_Baseline_PCA.py`

---

## 11. Analysis & Evaluation

Once embeddings are generated, several analyses are performed.

### 11.1 Distance Analysis

**Files:** `Experiments/ExecuteExperiments/Helpers/CDistanceAnalysis.py`, `CDistanceFunctions.py`

Two distance metrics are computed between any pair of user embeddings:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Cosine Distance | `1 - cosine_similarity(v1, v2)` | 0 = identical direction, 1 = orthogonal |
| Euclidean Distance | `‖v1 - v2‖₂` | Straight-line distance in embedding space |

**Key analysis:** Compare distances between canary users:
- Canary 1 users (exact duplicates) should have distance ≈ 0
- Canary 2 users (slight variants) should have small but non-zero distance
- Random user pairs should have much larger distances

This validates that the embedding captures meaningful behavioral similarity.

### 11.2 Top-K Similarity Search

**File:** `Experiments/ExecuteExperiments/Helpers/CTopKUsers.py`

For a query user, finds the K most similar users by:
1. Computing distances to all other users
2. Sorting in ascending order (smallest distance = most similar)
3. Returning the top K user IDs

Supports comparison across multiple algorithm IDs to find users that appear in the top-K across *all* algorithms (consensus similar users).

### 11.3 Clustering Analysis

**File:** `Experiments/ExecuteExperiments/Helpers/CClusteringAnalysis.py`

Applies K-Means clustering to the embedding matrix:

1. Try K from 2 to 10 clusters
2. Compute WCSS (Within-Cluster Sum of Squares) for each K
3. Use `KneeLocator` (from the `kneed` library) to find the "elbow point" — the K where adding more clusters gives diminishing returns
4. Run final K-Means with the optimal K
5. Output: cluster label per user, cluster centroids

### 11.4 PCA Projection for Visualization

**File:** `Experiments/ExecuteExperiments/Helpers/CPCAAnalysis.py`

Projects the learned embeddings (8D) down to 2D using PCA for scatter plot visualization. This lets you visually inspect whether similar users cluster together.

---

## 12. Visualization & Plots

**Entry point:** `Experiments/ExecuteExperiments/PlotExperimentalData.py`

**Orchestrator:** `Experiments/Plots/CPlotExperimentalData.py`

Running the plot entry point generates all plots for a given algorithm. Each plot type is implemented in a dedicated file under `Experiments/Plots/`.

### Plots Generated (per algorithm)

| Plot | What It Shows | File |
|------|--------------|------|
| **Canary Distance (Euclidean)** | Distance between canary and reference user pairs | `CPlotDistance.py` |
| **Canary Distance (Cosine)** | Cosine distance for canary validation | `CPlotDistance.py` |
| **All Pair Distances** | Histogram of distances across all user pairs | `CPlotDistance.py` |
| **Training Loss** | Scatter plot of per-user convergence loss | `CPlotExperimentalData.py` |
| **PCA 2D Projection** | 2D scatter of all users' embeddings | `CPlotCommon.py` |
| **PCA with Canary Highlighted** | Same 2D projection, canary users marked | `CPlotCommon.py` |
| **Clustering (Elbow Curve)** | WCSS vs. K to show optimal cluster count | `CPlotExperimentalData.py` |
| **Cluster Centroids in PCA** | Centroid positions overlaid on 2D projection | `CPlotExperimentalData.py` |

---

## 13. Running the Code

### Prerequisites

- Virtual environment activated (see [Section 3](#3-setup-virtual-environment))
- All commands run from the **project root directory** (`UserEmbeddings_MCP/`)

### Step 1 (Optional): Regenerate Synthetic Data

> Skip this step if `Experiments/Data/mcp_interactions_u10000.db` already exists.

```bash
python Experiments/DataGeneration/GenerateSyntheticData.py
```

### Step 2: Run Embedding Algorithms

This runs Algorithm 1 (data prep) then Algorithms 2 and 3 and saves results to `Experiments/Data/ExperimentResults/`.

```bash
python Experiments/ExecuteExperiments/RunExperiments_Algorithms.py
```

**Expected output files:**
```
Experiments/Data/ExperimentResults/
├── user_embeddings_alg_2.pt     # Algorithm 2 embeddings (PyTorch tensor)
├── training_loss_alg_2.pkl      # Algorithm 2 per-user loss (Pickle)
├── user_embeddings_alg_3.pt     # Algorithm 3 embeddings
└── training_loss_alg_3.pkl      # Algorithm 3 per-user loss
```

### Step 3 (Optional): Run PCA Baseline

```bash
python Experiments/ExecuteExperiments/RunExperiments_Baseline_PCA.py
```

### Step 4: Generate Plots

```bash
python Experiments/ExecuteExperiments/PlotExperimentalData.py
```

This generates all analysis plots for Algorithms 2 and 3.

```bash
# For PCA baseline plots:
python Experiments/ExecuteExperiments/PlotBaselineClustering.py
```

---

## 14. End-to-End Data Flow

```
1. SQLite Database
   └── Tables: users, sessions, session_interactions, mcp_tools, canary_users

2. Algorithm 1 (Alg_1_DataPreparation.py)
   └── Reads DB via CDataPreparationHelper
   └── Produces: all_C_hat_u_1 = { user_id: { tool_id: norm_freq } }

3. CDataMain (Algorithms/Helpers/CDataMain.py)
   └── Converts all_C_hat_u_1 → PyTorch tensor MAT_u_tau
       Shape: (num_users, num_tools)
       Fill value 1e-4 for missing (unused) tools

4. Algorithm 2 / 3
   └── Reads MAT_u_tau using the strategy required by each algorithm
   └── Produces: MAT_E tensor, shape (num_users, embedding_dim)
   └── Produces: loss_for_each_user list

5. CResultsStore (Experiments/ExecuteExperiments/Helpers/CResultsStore.py)
   └── Saves MAT_E to user_embeddings_alg_{N}.pt
   └── Saves losses to training_loss_alg_{N}.pkl

6. Analysis (CDistanceAnalysis, CClusteringAnalysis, CTopKUsers, CPCAAnalysis)
   └── Loads .pt file, runs analysis, prints results

7. Plotting (CPlotExperimentalData, CPlotDistance, CPlotCommon)
   └── Generates matplotlib figures
```

---

## 15. Important File Reference

| File | Purpose | When to Edit |
|------|---------|-------------|
| `Experiments/CConfig.py` | All global constants | Scale up/down experiment size, change embedding dimensions |
| `Algorithms/Alg_1_DataPreparation.py` | Normalization algorithm | Change how tool usage is normalized |
| `Algorithms/Alg_2_AutoEncoder.py` | Shared autoencoder training | Change training parameters |
| `Algorithms/Alg_3_MatrixFactorization.py` | Truncated-SVD embedding algorithm | Change matrix factorization behavior |
| `Algorithms/Helpers/CDataMain.py` | Sparse dict → tensor conversion | Change fill value or indexing logic |
| `Algorithms/Helpers/CUserToolAutoencoder.py` | Shared autoencoder model | Change encoder or decoder architecture |
| `Algorithms/Helpers/CSingleLayer.py` | Legacy single-layer model | Change the legacy network architecture |
| `Algorithms/Helpers/CModelTraining.py` | Legacy model training loop | Change legacy optimizer, loss, or epochs |
| `Algorithms/Helpers/CPolynomialFitReduction.py` | Polynomial fitting | Change polynomial fitting strategy |
| `Experiments/Database/CDatabaseManager.py` | DB query interface | Add new queries or tables |
| `Experiments/Database/CSQLLite.py` | Low-level SQLite wrapper | Change DB performance settings |
| `Experiments/DataGeneration/CGenerateSyntheticData.py` | Data generator | Change how synthetic sessions are created |
| `Experiments/ExecuteExperiments/RunExperiments_Algorithms.py` | Main experiment runner | Change which algorithms to run |
| `Experiments/ExecuteExperiments/PlotExperimentalData.py` | Plot orchestrator | Change which plots to generate |
| `Experiments/ExecuteExperiments/Helpers/CResultsStore.py` | File I/O for embeddings | Change file format or location |
| `Experiments/ExecuteExperiments/Helpers/CDistanceFunctions.py` | Distance metric formulas | Add new distance metrics |
| `Experiments/ExecuteExperiments/Helpers/CClusteringAnalysis.py` | K-Means + elbow detection | Change clustering strategy |

---

## 16. Key Design Decisions

### Canary Users for Validation

Since ground truth for user similarity does not exist in real data, the synthetic dataset includes "canary" users — known-identical or near-identical behavioral copies. If the embedding algorithm is working correctly:
- **Canary 1** (exact copies): embedding distance ≈ 0
- **Canary 2** (slight variants): embedding distance > 0 but small
- **Random pairs**: embedding distance >> 0

This provides a built-in, quantitative correctness check for any new algorithm.

### Sparse Dictionary Representation

The user-tool matrix is stored as a dictionary of dictionaries rather than a dense 2D array because:
- There are ~10,000 tools total but each user interacts with only a small subset
- Storing zeros for every unused tool would waste significant memory
- The sparse format is converted to a dense tensor only when needed for computation

### Shared and Per-User Training

Algorithms 2 and 3 learn shared coordinate systems using an autoencoder and
matrix factorization, respectively.

### ID Offset Convention

SQL databases use 1-indexed IDs (users, tools start at 1). PyTorch tensors use 0-indexed arrays. Throughout the code, the conversion `tensor_index = db_id - 1` is applied wherever database IDs are used to index tensors.

### Database Performance Tuning

`Experiments/Database/CSQLLite.py` enables several SQLite performance optimizations:
- **WAL mode** (Write-Ahead Logging): Allows concurrent reads during writes
- **Cache size 40,000 pages (~40MB)**: Keeps frequently accessed pages in memory
- **Temp store in MEMORY**: Sorts and indices done in RAM, not disk
- **Synchronous = NORMAL**: Balances write safety with speed

These are critical when generating 10,000 users' worth of interaction data (millions of rows).
