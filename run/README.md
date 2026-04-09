# Run Workspace

This folder is the Colab-friendly experiment workspace for the current `docs/main.tex` methodology.

It holds:

- `colab_full_experiment.ipynb`: thin Session 1 notebook for the primary `MODE="auto"` run
- `colab_worker_session.ipynb`: thin Session 2+ notebook for one extra `MODE="worker"` GPU session
- `cache/`: cached RecBole benchmark data, checkpoints, and optional carbon outputs
- `results/`: score files, reranking metrics, manifests, and summary CSVs
- `figures/`: paper-ready plots referenced from the Results section
- `logs/`: one worker log per `(category, model)` job

The main runner is still:

```bash
python scripts/05_run_full_experiment.py --run-dir run
```

The Colab helper runner is now:

```bash
python scripts/06_colab_session.py --project-root . --mode auto --install --verify-runtime
```

For a single-machine run, that command will:

1. audit the interim timestamp splits,
2. build cached RecBole benchmark files under `run/cache/recbole/`,
3. train `BPR`, `NeuMF`, and `LightGCN`,
4. rerank with the 16-point `lambda` grid from `docs/main.tex`,
5. evaluate the Pareto trade-offs, and
6. generate the paper plots into `run/figures/`.

For parallel Colab GPU runs, the notebooks are intentionally minimal:

1. Open `colab_full_experiment.ipynb` in the first Colab GPU session and leave `MODE = 'auto'`.
2. Open `colab_worker_session.ipynb` in exactly one additional Colab GPU session when you want a second worker.
3. Run all cells in each notebook.
4. Let `scripts/06_colab_session.py` handle repo sync, dependency checks, batch-size defaults, and mode dispatch.
5. Let each worker claim jobs from the shared `run/results/_job_state/` directory.

The parallel flow is safe because `scripts/05_run_full_experiment.py` atomically prepares shared inputs, claims pending `(category, model)` jobs, and finalizes plots once when the last worker finishes.

For a quick smoke run, pass `--max-users 5000 --categories electronics --models BPR`.
