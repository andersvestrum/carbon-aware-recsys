# Presentation Outline: Trading Engagement for Sustainability

**CS294 | UC Berkeley**
**~20 minutes**

All figures referenced below are in `docs/figures/`. All numbers come from `docs/main.tex` and `output/results/`.

---

## Deck Structure

---

### Slide 1. Title

**Trading Engagement for Sustainability: Carbon-Aware Re-ranking for E-commerce Recommendations**

- Names, course, date
- One line below the title: *Can we reduce the carbon footprint of e-commerce recommendations without materially harming recommendation quality?*

---

### Slide 2. The Problem

**Two sentences on the board:**

- Recommender systems determine which products millions of people see. That choice has a carbon consequence.

**Three bullets:**

- Global e-commerce sales: USD 27 trillion in 2022 (UNCTAD)
- Environmental impact varies with what gets recommended, not just how goods are shipped
- Product Carbon Footprint (PCF) labels are missing for almost all catalog items at scale

**Speaker note:** Keep this grounded. The point is that the ranking algorithm is not environmentally neutral — it is a form of consumption infrastructure.

---

### Slide 3. Research Questions

Three questions, one per line:

> **RQ1.** How does carbon-aware re-ranking affect recommendation quality across models?
>
> **RQ2.** How much carbon reduction is achievable before quality degrades?
>
> **RQ3.** Do different recommendation models expose different engagement and carbon trade-offs?

**Below the questions, one sentence:**
Main approach: estimate PCF for every catalog item, then re-rank recommendation candidates using an explicit, tunable weight between engagement and carbon.

---

### Slide 4. Pipeline Overview

**Suggested visual:** compile and export `docs/flow_diagram.tex`, or draw a clean linear diagram.

**Five stages across the slide:**

```
Carbon Catalogue (866 LCA products)
        +
Amazon metadata (72,000 products)
        |
        v
PCF Estimation
(few-shot LLM + neighbour fallback)
        |
        v
RecBole Candidate Generation
(BPR / NeuMF / LightGCN, top-100 per user)
        |
        v
Carbon-Aware Re-ranking
(lambda sweep, top-10 returned)
        |
        v
Evaluation
(NDCG@10, Recall@10, AvgPCF@10)
```

**Speaker note:** This slide should orient the audience for everything that follows. Spend time here.

---

### Slide 5. Estimating PCF at Catalog Scale

**Left column — the challenge:**

- No LCA data exists for most Amazon products
- Source: Carbon Catalogue — 866 products with measured PCF (kg CO2e)
- Encoder: `all-MiniLM-L6-v2`, cosine similarity, 5 nearest neighbours

**Right column — three methods compared:**

| Method | Description |
|--------|-------------|
| Neighbour average | Mean PCF of 5 nearest catalogue neighbours |
| Zero-shot LLM | Qwen-2.5-3B prompted with title only |
| Few-shot LLM | Qwen-2.5-3B with 5 labelled examples + chain-of-thought |

**Speaker note:** The three-method comparison is methodologically important — it shows we are not just using LLM outputs blindly. We evaluate before we deploy.

---

### Slide 6. PCF Estimation Results

**Suggested visual:** `docs/figures/pcf_subset_rmse.png` and `docs/figures/pcf_subset_spearman.png` side by side, or the four-panel figure.

**Table on slide (consumer-scale subset, n = 771):**

| Method | RMSE | MAE | Median AE | Spearman |
|--------|------|-----|-----------|----------|
| Neighbour average | 3,002.0 | 959.4 | 123.3 | **0.728** |
| Zero-shot LLM | 1,712.8 | 790.7 | 93.2 | 0.064 |
| Few-shot LLM | **1,708.6** | **695.1** | **58.6** | 0.518 |

**Key point:** Few-shot LLM wins on absolute error; neighbour average wins on rank preservation. The downstream pipeline uses few-shot when available (99.98% of 72,000 products), falling back to neighbour average otherwise.

**Speaker note:** Do not overstate this. The point is that the signal is meaningful enough to support a principled downstream trade-off analysis, not that PCF prediction is solved.

---

### Slide 7. Re-ranking Formulation

**Centre of slide, displayed formula:**

$$s_{ui} = (1 - \lambda)\,\tilde{y}_{ui} \;-\; \lambda\,\widetilde{\mathrm{PCF}}_i$$

**Below the formula:**

- $\tilde{y}_{ui}$: per-user min-max normalised relevance score from RecBole
- $\widetilde{\mathrm{PCF}}_i$: global min-max normalised carbon footprint
- $\lambda = 0$: pure engagement ranking
- $\lambda = 1$: pure carbon minimisation
- Sweep 25 values of $\lambda$ from 0 to 1, evaluate top-10 list at each point

**One line at the bottom:** The sustainability weight $\lambda$ is explicit and inspectable — it does not disappear into model weights.

**Speaker note:** This is the core method slide. It is deliberately simple. The transparency argument (explicit lambda vs. hidden objective) is worth saying out loud.

---

### Slide 8. Experimental Setup

**Three columns:**

| | |
|---|---|
| **Dataset** | Amazon Reviews 2023 |
| **Categories** | Electronics, Home and Kitchen, Sports and Outdoors |
| **Users per category** | 1,000 (approx. 6,400 to 6,550 interactions each) |
| **Models** | BPR (Rendle 2009), NeuMF (He 2017), LightGCN (He 2020) |
| **Candidate pool** | Top-100 from RecBole per user |
| **Final list size** | Top-10 after re-ranking |
| **Lambda grid** | 25 points: dense near 0 and 0.9 to 1.0 |
| **Engagement metrics** | NDCG@10, Recall@10 |
| **Carbon metric** | AvgPCF@10 (kg CO2e, mean over top-10 list) |

---

### Slide 9. Lambda Sensitivity: Plateau Then Cliff

**Suggested visual:** `docs/figures/lambda_sensitivity_home_and_kitchen_NeuMF.png` (shows the positive result clearly) alongside `docs/figures/lambda_sensitivity_electronics_NeuMF.png` (shows the early-cliff exception). Or use one category with all three models as panels.

**Key points on slide:**

- For BPR and LightGCN: NDCG@10 stays nearly flat until lambda reaches 0.91 to 0.94
- For NeuMF: cliff arrives much earlier in Electronics (lambda = 0.30), later in Home and Kitchen (lambda = 0.96)
- AvgPCF@10 falls steadily across the plateau in all cases

**One line:** The engagement cost and carbon benefit are not symmetric across models.

---

### Slide 10. Pareto Frontiers

**Suggested visual:** `docs/figures/multimodel_pareto_electronics.png` — shows all three model curves overlaid, upper-left is better.

**Key points:**

- Each point on the curve is one lambda value
- NeuMF starts highest in engagement (NDCG@10 = 0.0618 in Electronics) but its frontier collapses early
- BPR and LightGCN start lower but maintain engagement across a far wider carbon range
- The frontier shape, not just the endpoint, determines practical usability

**Speaker note:** This is the core visualisation for RQ3. Let the figure speak; annotate the BPR and NeuMF curves directly if possible.

---

### Slide 11. Summary: Maximum Carbon Reduction Within 5% NDCG Budget

**Suggested visual:** `docs/figures/carbon_reduction_heatmap.png`

**Reading the heatmap:**
- Each cell: maximum carbon reduction achievable while losing at most 5% of baseline NDCG@10
- Green = more carbon reduction available

**Highlight three findings:**

1. Home and Kitchen is the most carbon-flexible category: all three models exceed 80% reduction (BPR 84.1%, LightGCN 83.7%, NeuMF 81.1%)
2. Electronics NeuMF is the hardest case: only 7.6% reduction within the 5% NDCG budget, because the engagement cliff arrives at lambda = 0.30
3. BPR and LightGCN consistently offer large headroom across all categories (69% to 86%)

**Speaker note:** This heatmap is the single most compact summary of the experiment. It directly answers RQ2 and RQ3 together.

---

### Slide 12. Best Pareto Operating Points

**Suggested visual:** `docs/figures/best_pareto_summary_bar.png`

**Or a compact table (subject to >= 10% carbon reduction):**

| Category | Model | NDCG@10 | Carbon reduction | lambda* |
|----------|-------|---------|-----------------|---------|
| Electronics | BPR | 0.0078 | 73.2% | 0.70 |
| Electronics | LightGCN | 0.0249 | 40.2% | 0.25 |
| Electronics | NeuMF | 0.0562 | 10.0% | 0.50 |
| Home and Kitchen | BPR | 0.0035 | 54.0% | 0.60 |
| Home and Kitchen | LightGCN | 0.0066 | 52.3% | 0.60 |
| Home and Kitchen | NeuMF | 0.0164 | 50.6% | 0.80 |
| Sports and Outdoors | BPR | 0.0036 | 12.9% | 0.075 |
| Sports and Outdoors | LightGCN | 0.0101 | 17.9% | 0.10 |
| Sports and Outdoors | NeuMF | 0.0101 | 10.9% | 0.25 |

**Speaker note:** Point out that Recall@10 tracks NDCG@10 closely at every operating point. The re-ranker changes which low-carbon items enter the list, not the overall recall structure.

---

### Slide 13. What We Found

Four findings, plainly stated:

1. Substantial carbon reduction is achievable before engagement degrades, for models with flat score distributions (BPR, LightGCN).
2. NeuMF has the highest absolute engagement but the tightest carbon headroom in high-variance categories like Electronics.
3. Home and Kitchen offers the most uniform flexibility: every model achieves over 80% carbon reduction within a 5% NDCG budget.
4. The engagement and carbon trade-off is shaped jointly by the recommendation model and the category's PCF distribution, not by either alone.

**One sentence at the bottom:** Post-hoc re-ranking with an explicit lambda is a practical, auditable mechanism for adding sustainability objectives to an existing recommendation system without retraining.

---

### Slide 14. Limitations and Future Directions

**Limitations:**

- PCF values are predicted, not measured via life-cycle assessment — estimation errors propagate downstream
- Offline evaluation cannot capture how real users respond to sustainability signals
- Amazon reviews are an imperfect proxy: selection bias, missing negatives, temporal confounds
- Dataset limited to 1,000 users per category; larger runs may shift results

**Future directions:**

- Online experiments to measure real behavioral response to carbon-aware rankings
- Multi-objective training rather than post-hoc re-ranking
- Improving PCF estimation coverage and precision
- Investigating user acceptance of explicit sustainability explanations

---

### Slide 15. Questions

Title of the project, names, repository link.

---

## Timing Guide

| Slides | Content | Time |
|--------|---------|------|
| 1 to 3 | Motivation and research questions | 3 min |
| 4 to 7 | Pipeline, PCF estimation, formulation | 6 min |
| 8 to 12 | Setup, results, heatmap, table | 8 min |
| 13 to 15 | Findings, limitations, questions | 3 min |

## If You Need to Cut to 10 Slides

- Merge slides 2 and 3 into one motivation and question slide
- Drop slide 8 (setup can be covered verbally)
- Merge slides 9 and 10 (show one lambda sensitivity plot and one Pareto frontier)
- Merge slides 13 and 14

## Key Figures

| Figure | File |
|--------|------|
| Pipeline diagram | `docs/flow_diagram.tex` |
| PCF estimation accuracy | `docs/figures/pcf_subset_rmse.png`, `docs/figures/pcf_subset_spearman.png` |
| Lambda sensitivity | `docs/figures/lambda_sensitivity_electronics_NeuMF.png`, `docs/figures/lambda_sensitivity_home_and_kitchen_NeuMF.png` |
| Pareto frontier (multi-model) | `docs/figures/multimodel_pareto_electronics.png` |
| Heatmap summary | `docs/figures/carbon_reduction_heatmap.png` |
| Best Pareto bar | `docs/figures/best_pareto_summary_bar.png` |

## Thesis Statement

We show that a simple post-hoc re-ranking strategy, parameterised by a single explicit weight, can achieve large reductions in the carbon footprint of e-commerce recommendations while preserving most of their engagement quality, and that the achievable trade-off depends jointly on the recommendation model and the category-specific distribution of product carbon footprints.
