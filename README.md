# Freight Spot Rate Prediction

Predict truckload spot rates for a forward two-month window, trained on
48,000 historical loads spanning January through October 2025.

## Results

| Metric | Value |
|--------|-------|
| Validation MAE (Sep-Oct fold) | ~$89 |
| Validation MedAPE (Sep-Oct fold) | ~1.0% |
| Validation RMSE | ~$629 |
| Validation predictions | 12,000 rows |
| December predictions | 31 rows |

## Methodology

### Target Formulation

Freight spot rates decompose as:

    posted_rate = distance * quote_signal * alpha

Training directly on raw `posted_rate` forces the model to rediscover the
known `distance * quote_signal` baseline from scratch. Instead, we isolate
the residual multiplier:

    alpha = posted_rate / (distance * quote_signal)

and reconstruct the prediction at inference:

    predicted_rate = distance * quote_signal * alpha_hat

This substitution reduces residual MAE from ~$246 (raw rate target) to ~$89,
eliminates the exp() underestimation bias of log-rate models, and produces a
well-behaved target with mean near 1.0.

### Model

`sklearn.ensemble.HistGradientBoostingRegressor` with `loss='absolute_error'`,
`max_iter=300`, `random_state=42`. L1 loss is required because the data
contains an ~11x rate spike regime concentrated in short-haul loads
(sub-150-mile) that pegs RMSE at ~$630 regardless of model complexity.
L1 optimises median error, achieving MedAPE ~1.0% out-of-sample.

Predicted alpha is clipped to [0.5, 3.5] before back-transforming to rates.

### Features

| Feature | Description |
|---------|-------------|
| `distance` | Haul length in miles |
| `log_distance` | Natural log of distance (linearises variance) |
| `inv_distance` | 1/distance (non-linear rate-per-mile decay) |
| `quote_signal` | Market rate index from broker API |
| `is_short_haul` | 1 if distance < 450 mi (spike regime indicator) |
| `weight_clean` | Payload in lbs (abs-corrected for telematics sign flips) |
| `day_of_week` | 0=Monday ... 6=Sunday |
| `equipment_Reefer` | Temperature-controlled premium |
| `equipment_Flatbed` | Open-deck premium |

**Dropped features:** `market_index` (15% regime drift between train and
validation periods causes systematic overestimation); lat/lon coordinates
(distance already encodes geography); day-of-year harmonics (Jan-Oct fit
cannot extrapolate to Nov-Dec phase).

### Data Cleaning

- **Negative weights (292 rows):** `weight_clean = abs(weight)` — telematics
  systems invert sign on compression events; magnitude is always valid.
- **Missing weights (300 rows):** Stratifed median imputation by equipment type.

### Temporal Validation

Rolling-origin splits prevent forward leakage:
- Fold 1: Train Jan-Jun, test Jul-Aug
- Fold 2: Train Jan-Aug, test Sep-Oct (most representitive of deployment window)

### December Forecast

The Lexington-Fort Wayne (360 mi, Dry Van, 32,000 lb) December chart uses:
1. Daily mean `quote_signal` across all equipment types in the validation set
   (163+ loads/day for a stable signal).
2. 7-day rolling mean with `min_periods=1` to smooth day-to-day noise.
3. Same trained model for prediction.

December rates are stable around $762-796, consistent with the calm
market environment in the November-December period
(market_index mean 0.93 vs 1.08 in the training window).

## Project Structure

```
.
├── src/
│   ├── data.py           # Loading, weight cleaning, schema validation
│   ├── features.py       # Feature engineering
│   ├── validate.py       # Rolling-origin temporal validation
│   └── train_predict.py  # Full fit, validation inference, December forecast
├── train-test.csv
├── validation.csv
├── december-chart-inputs.csv
├── validation_predictions.csv
├── score.py
├── requirements.txt
└── README.md
```

## Reproducing Results

```bash
# Set up environment
uv venv .venv
uv pip install -r requirements.txt scikit-learn

# Optional: inspect rolling-origin metrics first
python -m src.validate

# Generate all predictions
python -m src.train_predict

# Run scorer
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions december-chart-inputs.csv
```

Expected output:
```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```
