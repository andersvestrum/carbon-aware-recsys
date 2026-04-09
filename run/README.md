# Run Workspace

This folder is the Colab-friendly experiment workspace for the current `docs/main.tex` methodology.

It holds:

- `colab_full_experiment.ipynb`: shared Colab notebook with a default `auto` mode that mounts Drive, clones or updates the repo, prepares caches, claims jobs, and finalizes plots
- `cache/`: cached RecBole benchmark data, checkpoints, and optional carbon outputs
- `results/`: score files, reranking metrics, manifests, and summary CSVs
- `figures/`: paper-ready plots referenced from the Results section
- `logs/`: one worker log per `(category, model)` job

The main runner is still:

```bash
python scripts/05_run_full_experiment.py --run-dir run
```

For a single-machine run, that command will:

1. audit the interim timestamp splits,
2. build cached RecBole benchmark files under `run/cache/recbole/`,
3. train `BPR`, `NeuMF`, and `LightGCN`,
4. rerank with the 16-point `lambda` grid from `docs/main.tex`,
5. evaluate the Pareto trade-offs, and
6. generate the paper plots into `run/figures/`.

For parallel Colab GPU runs, the notebook is now designed so each session can usually just press "Run all":

1. Open the notebook in one or more Colab GPU sessions.
2. Leave `MODE = 'auto'` unless you specifically want `prepare`, `worker`, or `finalize` only.
3. Let the notebook clone or update `carbon-aware-recsys` in `MyDrive/` automatically.
4. Run all cells in each worker session.
5. Let each worker claim jobs from the shared `run/results/_job_state/` directory.

The parallel flow is safe because `scripts/05_run_full_experiment.py` atomically prepares shared inputs, claims pending `(category, model)` jobs, and finalizes plots once when the last worker finishes.

For a quick smoke run, pass `--max-users 5000 --categories electronics --models BPR`.
