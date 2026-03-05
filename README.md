# Carbon-Aware Recommender System

A multi-objective recommender system that balances engagement with carbon footprint awareness. Uses RecBole for candidate generation, carbon-aware re-ranking, and DeepFM for engagement prediction.

## Pipeline

1. **Candidate Generation** — RecBole (BPR, NeuMF, SASRec, LightGCN) generates top-K items per user
2. **Carbon-Aware Re-ranking** — `score = relevance - λ * carbon_footprint`
3. **Engagement Prediction** — DeepFM predicts engagement probability for re-ranked items
4. **Evaluation** — Compare engagement vs carbon footprint trade-offs

## Data Sources

- **User interactions**: [Amazon Review Data v2](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) — verified purchases with ratings as engagement proxy
- **Carbon footprints**: [The Carbon Catalogue](https://www.kaggle.com/datasets/jeannettesavage/the-carbon-catalogue-public-database) — product-level carbon data + prediction for unseen products

## Project Structure

```
carbon-aware-recsys/
├── configs/                    # Experiment & model configuration files
│   ├── recbole/                #   RecBole model configs (BPR, SASRec, etc.)
│   ├── deepfm/                 #   DeepFM training configs
│   └── reranking/              #   Re-ranking hyperparameters (λ values, etc.)
├── data/
│   ├── raw/                    # Original downloaded datasets (do not modify)
│   │   ├── amazon/             #   Amazon review data
│   │   └── carbon_catalogue/   #   Carbon Catalogue product-level data
│   ├── interim/                # Intermediate transformed data
│   └── processed/              # Final datasets ready for modeling
│       ├── recbole/            #   RecBole-formatted .inter/.item/.user files
│       └── carbon/             #   Carbon footprint lookup + predictions
├── notebooks/                  # Jupyter notebooks for exploration & analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_carbon_prediction.ipynb
│   └── 03_results_analysis.ipynb
├── src/                        # Source code
│   ├── data/                   #   Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── amazon_loader.py    #     Download & parse Amazon reviews
│   │   ├── carbon_loader.py    #     Download & parse Carbon Catalogue
│   │   └── preprocess.py       #     Clean, merge, format for RecBole
│   ├── carbon/                 #   Carbon footprint estimation
│   │   ├── __init__.py
│   │   └── predictor.py        #     Predict footprint for unseen products
│   ├── reranking/              #   Carbon-aware re-ranking module
│   │   ├── __init__.py
│   │   └── carbon_reranker.py  #     score = relevance - λ * carbon
│   ├── engagement/             #   DeepFM engagement prediction
│   │   ├── __init__.py
│   │   ├── deepfm.py           #     DeepFM model wrapper
│   │   └── train.py            #     Training script
│   └── evaluation/             #   Evaluation & metrics
│       ├── __init__.py
│       └── metrics.py          #     Engagement vs carbon trade-off metrics
├── scripts/                    # Runnable pipeline scripts
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   ├── 03_train_recommender.py
│   ├── 04_predict_carbon.py
│   ├── 05_rerank.py
│   ├── 06_predict_engagement.py
│   └── 07_evaluate.py
├── output/                     # Experiment outputs
│   ├── models/                 #   Saved model checkpoints
│   ├── results/                #   Evaluation results & tables
│   └── figures/                #   Plots & visualizations
├── references/                 # Papers, notes, related work
├── requirements.txt
└── README.md
```

## Key References

- [RecBole](https://recbole.io/) — Recommendation toolkit
- [DeepFM](https://arxiv.org/pdf/1703.04247) — CTR prediction model
- [The Carbon Catalogue](https://www.nature.com/articles/s41597-022-01178-9) — Product carbon footprints
- [Amazon Review Data](https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf) — Interaction dataset
- [Multi-stakeholder RecSys](https://arxiv.org/pdf/1907.13158) — Engagement vs environment
- [Transparent Recommendations](https://www.cs.cornell.edu/~tj/publications/schnabel_etal_20a.pdf)

## Limitations

- **Selection bias**: Only a fraction of purchases become reviews; reviewers are not representative
- **Missing negatives**: Lack of review ≠ dislike
- **Temporal confounds**: Popularity shifts, seasonality
- Amazon used instead of behavioral dataset (e.g., SYNERISE) because products must be identifiable for carbon footprint mapping
