# User Embedding Algorithms

This folder contains the active data-preparation and embedding algorithms used
by `Experiments/ExecuteExperiments/RunExperiments_Algorithms.py`.

## Active Algorithms

| ID | File | Method | Training |
|----|------|--------|----------|
| 1 | `Alg_1_DataPreparation.py` | Normalize user-tool interaction counts | None |
| 2 | `Alg_2_AutoEncoder.py` | Shared nonlinear autoencoder | Iterative |
| 3 | `Alg_3_MatrixFactorization.py` | Truncated singular value decomposition | Direct factorization |
| 11 | `Alg_Baseline_PCA.py` | Standardized PCA baseline | Direct projection |
| 21 | `Alg_Data_Raw.py` | Raw tool-count baseline | None |

Algorithms 2 and 3 return:

- `MAT_E`: a `torch.Tensor` with shape `(number_of_users, embedding_dimensions)`
- `loss_for_each_user`: a tensor containing one reconstruction loss per user

Both algorithms consume an `IUserToolMatrix` implementation whose
`get_MAT_u_tau()` method returns the complete user-tool matrix.

## Algorithm 2: Autoencoder

`Alg_2_AutoEncoder` trains one encoder and decoder across all users. The encoder
maps each complete tool-usage vector into a shared embedding space, while the
decoder reconstructs the original vector.

Current tuning:

| Parameter | Value |
|-----------|-------|
| Maximum epochs | 5,000 |
| Target weighted MSE | `5e-5` |
| Learning rate | `0.02` |
| Batch size | 256 |
| Optimizer | Adam |

Observed tool values receive more weight than near-zero values so that sparse
user activity is not overwhelmed by unused tools. Training stops early when the
average weighted reconstruction loss reaches the target.

## Algorithm 3: Matrix Factorization

`Alg_3_MatrixFactorization` uses deterministic `TruncatedSVD` to approximate:

```text
user_tool_matrix ~= user_embeddings x tool_factors
```

The transformed user factors become the embeddings. Per-user loss is the mean
squared error between the original and reconstructed tool-usage vectors.

The requested embedding dimension must not exceed:

```text
min(number_of_users, number_of_tools)
```

Unlike the autoencoder, matrix factorization has no epochs or learning rate. It
is faster and linear, while the autoencoder can represent nonlinear patterns.

## Baselines

- `Alg_Baseline_PCA.py` standardizes tool columns before applying PCA.
- `Alg_Data_Raw.py` preserves raw tool counts for comparison.

Baseline IDs are stored separately from active embedding IDs so analysis code
can compare learned embeddings against direct representations.

## Retired Implementations

`Unused/` contains retired algorithm implementations retained for historical
comparison. Experiment runners, plots, and analysis helpers do not import or
execute anything from that folder.

## Tests

From the `UserEmbeddings_MCP` folder:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  Algorithms.TestData.test_algorithm_2_autoencoder `
  Algorithms.TestData.test_algorithm_3_matrix_factorization
```

The tests verify output shapes, finite reconstruction losses, canary-user
similarity, deterministic autoencoder training, and matrix-rank validation.
