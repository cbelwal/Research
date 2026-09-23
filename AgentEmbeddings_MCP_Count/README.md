# AgentEmbeddings_MCP_Count

A machine learning research system that generates low-dimensional **agent embeddings** from agent interaction data with **MCP (Model Context Protocol)** tools. The system models agent behavior patterns through their tool usage and creates compact vector representations (embeddings) that enable agent similarity analysis, clustering, and recommendation tasks.

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

> **Can we represent an agent's behavior — specifically, which MCP tools they use and how often — as a short fixed-length vector (an "embedding"), and do similar agents end up with similar embeddings?**

### What is an MCP Tool?

MCP (Model Context Protocol) servers expose *tools* — callable functions that an AI assistant can invoke. A single MCP server might provide 1–50 tools (e.g., file reading, web search, database queries). Agents interact with these tools across many sessions.

### What is a Agent Embedding?

An embedding is a short, dense numeric vector (e.g., 8 numbers) that summarizes an agent's behavioral profile. Agents who tend to use the same tools in similar proportions should end up close together in embedding space.

### Why Does This Matter?

- **Agent Similarity:** Find agents with similar tool usage habits.
- **Clustering:** Group agents with similar behavior automatically.
- **Recommendations:** Suggest tools to agents based on similar agents' behavior.
- **Anomaly Detection:** Identify agents whose behavior deviates significantly from peers.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Synthetic Data Generation                  │
│   (Agents, MCP Servers, Tools, Sessions, Interactions)   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   SQLite Database                       │
│   mcp_interactions_a100000.db                           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│          Algorithm 1: Data Preparation                  │
│   Normalize tool usage frequencies → sparse matrix      │
│   Output: {agent_id: {tool_id: frequency}} dict          │
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
│  agent_embeddings_     │    │  stored separately      │
│  alg_N.pt             │    │                         │
└───────────┬───────────┘    └─────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│                  Analysis & Evaluation                  │
│  - Canary agent distance validation                      │
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
AgentEmbeddings_MCP_Count/
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
│       ├── IAgentToolMatrix.py           # Abstract interface for data matrices
│       ├── CDataMain.py                 # Converts dict → PyTorch tensor matrix
│       ├── CAgentToolAutoencoder.py      # Shared encoder/decoder model
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
│   │       ├── CTopKAgents.py            # Top-K most similar agents lookup
│   │       └── CPCAAnalysis.py          # 2D PCA projection for visualization
│   │
│   ├── Plots/
│   │   ├── CPlotExperimentalData.py     # Orchestrates which plots to generate
│   │   ├── CPlotCommon.py               # Shared matplotlib utilities
│   │   ├── CPlotDistance.py             # Distance histogram plots
│   │   └── CPlotSyntheticData.py        # Raw data distribution plots
│   │
│   └── Data/
│       ├── mcp_interactions_a100000.db  # Generated SQLite database (100,000 agents)
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
    MAX_AGENTS = 100000             # Total agents in the synthetic dataset
    DB_FILE_NAME = "mcp_interactions_a100000.db"  # Must match MAX_AGENTS

    MAX_MCP_SERVERS = 200          # Number of MCP servers in the simulation
    MAX_TOOLS_PER_MCP_SERVER = 30  # Max tools per server (min = 1)
    MIN_TOOLS_PER_MCP_SERVER = 1

    SESSIONS_PER_AGENT_MEAN = 100   # Average sessions per agent (normal dist)
    SESSIONS_PER_AGENT_STD  = 200
    MAX_SESSIONS_PER_AGENT = 500   # Hard upper bound
    SESSIONS_LENGTH_MEAN = 20      # Average sequence positions per session
    SESSIONS_LENGTH_STD  = 10
    MIN_TOOL_CALLS_PER_SEQUENCE = 0
    MAX_TOOL_CALLS_PER_SEQUENCE = 5

    EMBEDDING_DIMENSIONS = 8       # Size of the output embedding vector

    PROB_OF_TOOL_FROM_SAME_MCP = 0.33  # Probability of tool clustering behavior

    PERCENTAGE_AGENTS_CANARY_1 = 5  # % of agents who are exact canary duplicates
    PERCENTAGE_AGENTS_CANARY_2 = 5  # % of agents who are slight canary variants
```

> **Note:** If you change `MAX_AGENTS`, you must also update `DB_FILE_NAME` to match (and regenerate the database). Similarly, if you change `EMBEDDING_DIMENSIONS`, re-run the embedding algorithms.

---

## 6. Data: Synthetic Generation & Database

### Why Synthetic Data?

Real MCP interaction logs may not be publicly available. Synthetic data allows controlled experiments where we know *exactly* which agents should cluster together (via "canary agents"), enabling objective evaluation.

### Database Schema

The SQLite database (`mcp_interactions_a100000.db`) has 6 tables:

```
mcp_servers          mcp_tools              agents
──────────────       ─────────────────      ──────
id (PK)              id (PK)                id (PK)
no_of_tools          mcp_server_id (FK)
                     mcp_tool_id

sessions             session_interactions   canary_agents
──────────────       ─────────────────      ──────────────
id (PK)              id (PK)                id (PK)
agent_id (FK)         session_id (FK)        agent_id (FK)
session_depth        tool_id (FK)           canary_category
                     sequence_number
```

### How Synthetic Data is Generated

**File:** `Experiments/DataGeneration/CGenerateSyntheticData.py`

1. **MCP Servers & Tools:** Creates 200 MCP servers, each with 1–30 tools randomly.
2. **Agents:** Creates 100,000 agents.
3. **Sessions:** For each agent, draws the number of sessions from `Normal(mean=100, std=200)`, clamped to the inclusive range 1–500. Each session's depth is the number of sequence positions, drawn from `Normal(mean=20, std=10)` with a minimum of 1.
4. **Session Interactions:** For each session:
   - Draw 0–5 tool calls independently for each sequence position.
   - Store every call as its own row. Calls in the same position share a `sequence_number`.
   - For each tool call:
     - With probability 0.33 → pick a tool from the *same* MCP server as the previous tool (mimicking realistic tool clustering)
     - With probability 0.67 → pick any tool at random
5. **Canary Agents:** A key validation mechanism:
   - **Canary Category 1** (5% of agents): Exact copies of a reference agent. These agents have identical session data. Any good embedding algorithm must produce nearly identical embeddings for them.
   - **Canary Category 2** (5% of agents): Near-copies of the reference agent with session length reduced by 1. These should produce *similar but not identical* embeddings.

### Regenerating the Database

```bash
# From the project root (with .venv activated)
python Experiments/DataGeneration/GenerateSyntheticData.py
```

> **Warning:** Generating `mcp_interactions_a100000.db` can take significant time and disk space.

---

## 7. Algorithm 1: Data Preparation

**File:** `Algorithms/Alg_1_DataPreparation.py`

This is always the **first step**. It converts raw interaction counts from the database into a normalized sparse matrix suitable for embedding generation.

### What It Produces

A Python dictionary of dictionaries representing a **sparse agent-tool matrix**:

```
all_C_hat_a_1 = {
    agent_id_1: { tool_id_A: 0.45, tool_id_B: 0.30, tool_id_C: 0.25 },
    agent_id_2: { tool_id_A: 0.10, tool_id_D: 0.90 },
    ...
}
```

Each value is a normalized frequency between 0 and 1, summing to 1.0 for each agent.

### Normalization Steps (per agent)

**Step 1 — Count raw tool calls:**
```
C_a[tool_id] = number of times agent u called tool_id across all sessions
```

**Step 2 — Normalize by total tool calls:**
```
C_hat_a[tool_id] = C_a[tool_id] / total_tool_calls_by_agent_u
```

**Step 3 — Re-normalize to [0, 1] summing to 1:**
```
C_hat_a_1[tool_id] = C_hat_a[tool_id] / sum(C_hat_a.values())
```

> The double normalization ensures that the values are always in [0, 1] and sum to exactly 1, making them compatible with sigmoid activations used in Algorithm 2.

### Why Sparse?

Most agents only interact with a small fraction of the ~5,000 total tools available (100 servers × up to 50 tools each). Storing the full dense matrix would be wasteful; the sparse dictionary stores only non-zero entries.

---

## 8. Algorithm 2: Shared Autoencoder Embeddings

**File:** `Algorithms/Alg_2_AutoEncoder.py`

### Concept

A single autoencoder is trained across all agents. Its encoder compresses each complete agent-tool usage vector into a fixed-size embedding, and its decoder reconstructs the original vector.

### Architecture

```
Agent-tool vector → Linear/ReLU → embedding → Linear/ReLU/Linear/Sigmoid → reconstructed vector
```

- **Input:** One normalized vector containing every tool's usage frequency
- **Encoder output:** The agent embedding, shared in one coordinate system across agents
- **Decoder output:** A reconstruction with the same number of columns as the input
- **Loss:** Weighted mean squared error, with observed tools weighted more heavily

### Training

**File:** `Algorithms/Helpers/CAgentToolAutoencoder.py`

```
Initialize one shared encoder and decoder
Repeat up to 5,000 epochs:
    Read batches of complete agent-tool vectors
    Encode each vector into embedding_dim values
    Decode each embedding back into an agent-tool vector
    Update the shared model with weighted reconstruction loss
Run the trained encoder over every agent to produce MAT_E
```

### Key Files

| File | Role |
|------|------|
| `Algorithms/Helpers/CAgentToolAutoencoder.py` | Shared encoder and decoder network |
| `Algorithms/Alg_2_AutoEncoder.py` | Batched training and embedding extraction |
| `Algorithms/Helpers/CDataMain.py` | Converts `all_C_hat_a_1` dict → PyTorch tensor matrix |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| MAX_EPOCHS | 5,000 |
| MIN_TARGET_LOSS | 5e-5 |
| LEARNING_RATE | 0.02 |
| OPTIMIZER | Adam |
| LOSS FUNCTION | Weighted MSE |

### Output

- `MAT_E` — tensor of shape `(num_agents, embedding_dim)` — the embedding matrix
- `loss_for_each_agent` — list of final MSE loss per agent (used to evaluate convergence)

---

## 9. Algorithm 3: Matrix Factorization

**File:** `Algorithms/Alg_3_MatrixFactorization.py`

Algorithm 3 applies truncated singular value decomposition to the complete
agent-tool matrix:

`agent_tool_matrix ≈ agent_embeddings × tool_factors`

`TruncatedSVD` learns both factors globally. Its transformed agent factors are
the embeddings, and each agent's mean squared reconstruction error is returned
as that agent's loss. The method is deterministic, non-neural, and independent
of tool ordering. The embedding dimension cannot exceed the smaller matrix
dimension.

---

## 10. Baseline: PCA Embeddings

**File:** `Algorithms/Alg_Baseline_PCA.py`

### Concept

Principal Component Analysis (PCA) is a classical dimensionality reduction method. Instead of a per-agent model, it learns a *global* linear projection from the full agent-tool matrix.

### How It Works

1. Build the full dense agent-tool matrix (shape: `num_agents × num_tools`)
2. Apply `StandardScaler` to normalize each tool's usage across agents
3. Fit PCA with `n_components = embedding_dim` (e.g., 8)
4. Transform all agents' rows through the learned projection
5. Result: each agent has an 8-dimensional embedding

### Key Difference from Algorithms 2 and 3

| Aspect | Algorithms 2 and 3 | PCA Baseline |
|--------|-----------|-------------|
| Training | Shared across agents | Global |
| Method | Neural / analytical | Linear projection |
| New agents | Alg 2 encodes directly; Alg 3 requires refitting | Can project with the fitted PCA model |

**Entry point:** `Experiments/ExecuteExperiments/RunExperiments_Baseline_PCA.py`

---

## 11. Analysis & Evaluation

Once embeddings are generated, several analyses are performed.

### 11.1 Distance Analysis

**Files:** `Experiments/ExecuteExperiments/Helpers/CDistanceAnalysis.py`, `CDistanceFunctions.py`

Two distance metrics are computed between any pair of agent embeddings:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Cosine Distance | `1 - cosine_similarity(v1, v2)` | 0 = identical direction, 1 = orthogonal |
| Euclidean Distance | `‖v1 - v2‖₂` | Straight-line distance in embedding space |

**Key analysis:** Compare distances between canary agents:
- Canary 1 agents (exact duplicates) should have distance ≈ 0
- Canary 2 agents (slight variants) should have small but non-zero distance
- Random agent pairs should have much larger distances

This validates that the embedding captures meaningful behavioral similarity.

### 11.2 Top-K Similarity Search

**File:** `Experiments/ExecuteExperiments/Helpers/CTopKAgents.py`

For a query agent, finds the K most similar agents by:
1. Computing distances to all other agents
2. Sorting in ascending order (smallest distance = most similar)
3. Returning the top K agent IDs

Supports comparison across multiple algorithm IDs to find agents that appear in the top-K across *all* algorithms (consensus similar agents).

### 11.3 Clustering Analysis

**File:** `Experiments/ExecuteExperiments/Helpers/CClusteringAnalysis.py`

Applies K-Means clustering to the embedding matrix:

1. Try K from 2 to 10 clusters
2. Compute WCSS (Within-Cluster Sum of Squares) for each K
3. Use `KneeLocator` (from the `kneed` library) to find the "elbow point" — the K where adding more clusters gives diminishing returns
4. Run final K-Means with the optimal K
5. Output: cluster label per agent, cluster centroids

### 11.4 PCA Projection for Visualization

**File:** `Experiments/ExecuteExperiments/Helpers/CPCAAnalysis.py`

Projects the learned embeddings (8D) down to 2D using PCA for scatter plot visualization. This lets you visually inspect whether similar agents cluster together.

---

## 12. Visualization & Plots

**Entry point:** `Experiments/ExecuteExperiments/PlotExperimentalData.py`

**Orchestrator:** `Experiments/Plots/CPlotExperimentalData.py`

Running the plot entry point generates all plots for a given algorithm. Each plot type is implemented in a dedicated file under `Experiments/Plots/`.

### Plots Generated (per algorithm)

| Plot | What It Shows | File |
|------|--------------|------|
| **Canary Distance (Euclidean)** | Distance between canary and reference agent pairs | `CPlotDistance.py` |
| **Canary Distance (Cosine)** | Cosine distance for canary validation | `CPlotDistance.py` |
| **All Pair Distances** | Histogram of distances across all agent pairs | `CPlotDistance.py` |
| **Training Loss** | Scatter plot of per-agent convergence loss | `CPlotExperimentalData.py` |
| **PCA 2D Projection** | 2D scatter of all agents' embeddings | `CPlotCommon.py` |
| **PCA with Canary Highlighted** | Same 2D projection, canary agents marked | `CPlotCommon.py` |
| **Clustering (Elbow Curve)** | WCSS vs. K to show optimal cluster count | `CPlotExperimentalData.py` |
| **Cluster Centroids in PCA** | Centroid positions overlaid on 2D projection | `CPlotExperimentalData.py` |

---

## 13. Running the Code

### Prerequisites

- Virtual environment activated (see [Section 3](#3-setup-virtual-environment))
- All commands run from the **project root directory** (`AgentEmbeddings_MCP_Count/`)

### Step 1 (Optional): Regenerate Synthetic Data

> Skip this step if `Experiments/Data/mcp_interactions_a100000.db` already exists.

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
├── agent_embeddings_alg_2.pt     # Algorithm 2 embeddings (PyTorch tensor)
├── training_loss_alg_2.pkl      # Algorithm 2 per-agent loss (Pickle)
├── agent_embeddings_alg_3.pt     # Algorithm 3 embeddings
└── training_loss_alg_3.pkl      # Algorithm 3 per-agent loss
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
   └── Tables: agents, sessions, session_interactions, mcp_tools, canary_agents

2. Algorithm 1 (Alg_1_DataPreparation.py)
   └── Reads DB via CDataPreparationHelper
   └── Produces: all_C_hat_a_1 = { agent_id: { tool_id: norm_freq } }

3. CDataMain (Algorithms/Helpers/CDataMain.py)
   └── Converts all_C_hat_a_1 → PyTorch tensor MAT_a_tau
       Shape: (num_agents, num_tools)
       Fill value 1e-4 for missing (unused) tools

4. Algorithm 2 / 3
   └── Reads MAT_a_tau using the strategy required by each algorithm
   └── Produces: MAT_E tensor, shape (num_agents, embedding_dim)
   └── Produces: loss_for_each_agent list

5. CResultsStore (Experiments/ExecuteExperiments/Helpers/CResultsStore.py)
   └── Saves MAT_E to agent_embeddings_alg_{N}.pt
   └── Saves losses to training_loss_alg_{N}.pkl

6. Analysis (CDistanceAnalysis, CClusteringAnalysis, CTopKAgents, CPCAAnalysis)
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
| `Algorithms/Helpers/CAgentToolAutoencoder.py` | Shared autoencoder model | Change encoder or decoder architecture |
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

### Canary Agents for Validation

Since ground truth for agent similarity does not exist in real data, the synthetic dataset includes "canary" agents — known-identical or near-identical behavioral copies. If the embedding algorithm is working correctly:
- **Canary 1** (exact copies): embedding distance ≈ 0
- **Canary 2** (slight variants): embedding distance > 0 but small
- **Random pairs**: embedding distance >> 0

This provides a built-in, quantitative correctness check for any new algorithm.

### Sparse Dictionary Representation

The agent-tool matrix is stored as a dictionary of dictionaries rather than a dense 2D array because:
- There can be up to 6,000 tools, but each agent interacts with only a small subset
- Storing zeros for every unused tool would waste significant memory
- The sparse format is converted to a dense tensor only when needed for computation

### Shared and Per-Agent Training

Algorithms 2 and 3 learn shared coordinate systems using an autoencoder and
matrix factorization, respectively.

### ID Offset Convention

SQL databases use 1-indexed IDs (agents, tools start at 1). PyTorch tensors use 0-indexed arrays. Throughout the code, the conversion `tensor_index = db_id - 1` is applied wherever database IDs are used to index tensors.

### Database Performance Tuning

`Experiments/Database/CSQLLite.py` enables several SQLite performance optimizations:
- **WAL mode** (Write-Ahead Logging): Allows concurrent reads during writes
- **Cache size 40,000 pages (~40MB)**: Keeps frequently accessed pages in memory
- **Temp store in MEMORY**: Sorts and indices done in RAM, not disk
- **Synchronous = NORMAL**: Balances write safety with speed

These are critical when generating 100,000 agents' worth of interaction data.
