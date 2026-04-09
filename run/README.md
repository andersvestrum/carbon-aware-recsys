# Run Workspace

This folder is the Colab-friendly experiment workspace for the current `docs/main.tex` methodology.

It holds:

- `colab_full_experiment.ipynb`: shared Colab notebook for `prepare`, `worker`, and `finalize` modes
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

For parallel Colab GPU runs, use the notebook like this:

1. Open one Colab session with `MODE = 'prepare'` and run it once.
2. Open multiple Colab GPU sessions with `MODE = 'worker'`, giving each a different `WORKER_NAME`.
3. Point every session at the same repo directory on Google Drive.
4. Let each worker claim jobs from the shared `run/results/_job_state/` directory.
5. Run one final session with `MODE = 'finalize'` after all jobs are done.

The worker mode is safe to run in parallel because `scripts/05_run_full_experiment.py` now atomically claims pending `(category, model)` jobs and skips duplicates.

For a quick smoke run, pass `--max-users 5000 --categories electronics --models BPR`.
