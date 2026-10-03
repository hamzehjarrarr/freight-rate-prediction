# Freight Rate Prediction Challenge

Predicts `posted_rate` (USD) for 12,000 validation loads (Nov-Dec 2025) and a
31-day December forecast for one fixed lane (Lexington -> Fort Wayne, 360 mi,
Dry Van, 32,000 lb). All work is in `solution.ipynb`.

## Repository contents

| File | What it is |
|---|---|
| `solution.ipynb` | The full solution: exploration, data-quality fixes, validation, model, predictions. Outputs are saved in the notebook. |
| `requirements.txt` | Python dependencies (pandas, numpy, matplotlib, LightGBM, scikit-learn). |
| `score.py` | Spotter's provided scorer, unchanged. |
| `outputs/validation_predictions.csv` | Final predictions (`load_id,predicted_rate`, 12,000 rows). |
| `outputs/december_chart_inputs.csv` | The December input file with `predicted_rate` filled in. |
| `outputs/scorer_results/candidate_december.png` | December chart produced by `score.py`. |

The provided data files are not included (see "Run it").

## Approach

1. **Data quality**
   - 292 negative weights (same magnitudes as the positive ones) -> absolute value.
   - Missing `weight` (0.6%) is left missing (LightGBM handles it); missing
     `market_index` (0.8%) is covered by the day-level median `mi_day` in models
     C and D.
   - Weight capped at 47,500 (1,191 rows) and distance floored at 70 (48 rows):
     looks like clipping; left as is.
   - About 1.4% of training rates are corrupted (rate per mile roughly x3.4 or
     x0.27 of normal), spread evenly over equipment and months. Rows whose rate
     per mile is outside 0.5x-2.0x of the typical rate for their
     (equipment, distance band) are excluded from training only.
2. **Model**: LightGBM (L1 loss) on log(rate per mile); dollars = prediction x distance.
   Four feature sets are trained and their predictions averaged in log space:
   - A: distance, equipment, weight
   - B: A + `market_index` + `quote_signal`
   - C: A + `mi_day` (day-level median `market_index`)
   - D: A + `mi_day` + `quote_signal`
3. **Validation** (time-aware, because validation data is entirely after training):
   - Forward folds: train on the past, predict the next two months
     (May-Jun, Jul-Aug, Sep-Oct).
   - Leave-one-month-out: hold out each month, train on the other nine.
   - Metric: MAE in dollars on rows with uncorrupted labels.

| Model | Leave-one-month-out mean / worst | Forward folds mean / worst |
|---|---|---|
| A | 83.8 / 140.4 | 91.2 / 147.1 |
| B | 57.9 / 117.1 | 69.2 / 87.9 |
| C | 80.6 / 115.2 | 87.1 / 118.2 |
| D | 59.9 / 115.0 | 71.1 / 91.2 |
| **Average of all four (submitted)** | 63.1 / 93.1 | 68.9 / 72.3 |

The average was chosen as the safer option: models using `quote_signal` (B, D)
do worst on August, the only month whose `quote_signal` profile resembles
November-December, and model C fails when the relationship between
`market_index` and rates drifts (Sep-Oct). Averaging cut the worst-case error.
The blend was selected after seeing these results, so the numbers are slightly
optimistic.

## Run it

1. Put the provided files in the repository root, with their original names:
   `train-test.csv`, `validation.csv`, `validation-predictions-template.csv`,
   `december-chart-inputs.csv`.
2. Install the dependencies and open the notebook:

```bash
python -m pip install -r requirements.txt
python -m pip install notebook
jupyter notebook solution.ipynb      # then Run all (about 3-5 minutes)
```

   It also runs in Google Colab: upload the notebook and the provided files,
   then Run all.

3. The notebook writes `outputs/validation_predictions.csv` and the completed
   `outputs/december_chart_inputs.csv`, and its last cell runs the scorer. To run
   the scorer by hand:

```bash
python score.py --predictions outputs/validation_predictions.csv \
  --december-predictions outputs/december_chart_inputs.csv \
  --output-dir outputs/scorer_results
```

## Known limitations

- The overall rate level drifts month to month, and the link between
  `market_index` and rates is not stable over time (January and September have
  similar index values but rate levels about 6 points apart). The Nov-Dec level
  is therefore the largest uncertainty.
- December has no training data and the chart input has only a date, so the
  December line uses the day's market values taken from the validation loads and
  is nearly flat; treat its small day-to-day variation as weak evidence.
- Eight validation cities (about 12% of loads are on unseen lanes) never appear
  in training; they are handled only through distance, equipment and market
  features.
- The hidden evaluation metric is unknown; MAE on clean rows is used here. If the
  hidden labels contain the same ~1.4% corruption, any submission's error will
  be dominated by it.
