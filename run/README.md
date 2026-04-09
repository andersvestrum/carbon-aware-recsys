# Run Workspace

This folder is the Colab-friendly experiment workspace for the current `docs/main.tex` methodology.

It holds:

- `colab_full_experiment.ipynb`: notebook entry point for Google Colab
- `cache/`: cached RecBole benchmark data, checkpoints, and optional carbon outputs
- `results/`: score files, reranking metrics, manifests, and summary CSVs
- `figures/`: paper-ready plots referenced from the Results section
- `logs/`: one worker log per `(category, model)` job

The main runner is:

```bash
python scripts/05_run_full_experiment.py --run-dir run
```

That command will:

1. audit the interim timestamp splits,
2. build cached RecBole benchmark files under `run/cache/recbole/`,
3. train `BPR`, `NeuMF`, and `LightGCN`,
4. rerank with the 16-point `lambda` grid from `docs/main.tex`,
5. evaluate the Pareto trade-offs, and
6. generate the paper plots into `run/figures/`.

For a quick smoke run, pass `--max-users 5000 --categories electronics --models BPR`.
