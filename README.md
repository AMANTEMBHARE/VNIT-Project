# NDT-XGBoost: Boosting with Differentiable Soft Decision Trees

A from-scratch PyTorch implementation of a gradient-boosting framework that keeps XGBoost's second-order (gradient/Hessian) objective but replaces the hard-split trees with **soft (neural) decision trees** as weak learners. Benchmarked against XGBoost on the UCI Covertype dataset.


## Method

**Boosting loop.** The ensemble keeps raw class scores `F(x)` (7 per sample). Each round:

1. Compute `P = softmax(F)`, then `G = P - Y` and `H = P(1-P)`.
2. Train a fresh soft tree `f` with Adam on XGBoost's second-order objective:
   `Σ [ G·f(x) + ½·H·f(x)² ] + ½·λ·‖leaf values‖²`
3. Update `F ← F + η·f(x)`.

**Soft decision tree.**
- Depth `d` gives `2^d - 1` internal nodes and `2^d` leaves.
- Each internal node is a linear gate: `P(left | x) = σ(w·x + b)`.
- A sample reaches a leaf with probability equal to the product of gate probabilities along its path.
- Each leaf holds a learnable 7-dim vector (one score per class).
- Output: `f(x) = Σ_leaves P(leaf | x) · leaf_value`. Everything is differentiable.

**Differences from XGBoost**

| | XGBoost | NDT-XGBoost |
|---|---|---|
| Splits | Hard, axis-aligned | Soft, oblique (linear in all features) |
| Tree fitting | Greedy split search, closed-form leaves | Adam on all gates and leaves jointly (approximate) |
| Trees per round (multiclass) | K (one per class) | 1, with vector-valued leaves |
| Speed | Very fast | Much slower |

## Dataset

UCI Covertype: 581,012 samples, 54 features (10 numeric, 44 binary), 7 forest-cover classes. Classes 0 and 1 make up about 85% of the data.


## Results (test set, single seed)

| Model | Test acc | Macro-F1 | Train time |
|---|---|---|---|
| XGBoost (depth 4, 200 rounds, lr 0.1) | 78.75% | 0.7114 | 3.8 s |
| NDT-XGBoost (depth 4, 200 rounds, lr 0.1, 5 epochs) | 80.97% | 0.6707 | 82 s |
| NDT-XGBoost (depth 6, 200 rounds, lr 0.3, 10 epochs) | 84.91% | 0.7286 | 236 s |
| XGBoost tuned (depth 8, up to 600 rounds, early stopping) | 91.22% | 0.8684 | 19 s |


## Ablations (validation set, depth 4 and 100 rounds unless stated)

**Depth** (lr 0.1, 5 inner epochs)

| Depth | Val loss | Val acc | Val macro-F1 | Time |
|---|---|---|---|---|
| 2 | 0.5308 | 77.63% | 0.616 | 25.5 s |
| 3 | 0.4872 | 79.50% | 0.629 | 26.4 s |
| 4 | 0.4581 | 81.15% | 0.658 | 33.4 s |
| 6 | 0.4232 | 82.95% | 0.686 | 56.0 s |
| 8 | 0.4084 | 83.69% | 0.693 | 124.2 s |

**Learning rate and inner epochs** (depth 4)

| Config | Val loss | Val acc | Val macro-F1 | Time |
|---|---|---|---|---|
| lr = 0.05 | 0.4776 | 80.58% | 0.641 | 33.5 s |
| lr = 0.1 (base) | 0.4581 | 81.15% | 0.658 | 33.4 s |
| lr = 0.3 | 0.4436 | 81.58% | 0.672 | 35.3 s |
| inner epochs = 2 | 0.5258 | 77.66% | 0.619 | 15.1 s |
| inner epochs = 5 (base) | 0.4581 | 81.15% | 0.658 | 33.4 s |
| inner epochs = 10 | 0.4122 | 83.45% | 0.711 | 71.9 s |



## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install numpy scipy pandas scikit-learn matplotlib xgboost ipykernel
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Open `ndt_xgboost.ipynb`, select the `.venv` kernel, and run all cells. The dataset downloads automatically via `sklearn.datasets.fetch_covtype`. Runtime: about 5 minutes for the baselines, plus about 4 minutes for the best NDT run on a laptop CPU.
