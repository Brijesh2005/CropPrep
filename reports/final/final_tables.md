# Paper tables (reports/final)

## TABLE 1 — Dataset distribution
| split | coconut | pepper | total |
|---|---|---|---|
| train | 1232 | 308 | 1540 |
| val | 1172 | 293 | 1465 |
| test | 784 | 196 | 980 |

Raw R5.10 binary pool: 57,660 fields (coconut 56,863 / pepper 797). Balancing rule: coconut = 4x pepper per split (seed 42).

## TABLE 2 — Feature groups
| group | variables | source |
|---|---|---|
| Tabular environment | annual_rainfall_mm, dewpoint_c, temperature_c, relative_humidity_pct, elevation, slope, soil_clay_pct, soil_sand_pct, soil_organic_carbon, soil_ph, soil_moisture + leading-window aggregates | R5.7/R5.9/DK grids |
| Location (sensitivity) | lat, lon | survey GPS |
| Imagery (temporal) | ndvi, evi, ndwi, ndre, savi, s2_obs_count, kharif_ndvi, kharif_evi, kharif_ndwi, rabi_ndvi, rabi_evi, rabi_ndwi per grid year 2018..survey year | DK grid composites (KNN-IDW x5km) |
| Excluded | Crop_Extent, fractions, dominance, NPP, sat_* survey-frame stats, identity | R5.9/R5.10 contracts |

## TABLE 3 — Model configurations
| model | family | selection | notes |
|---|---|---|---|
| Tabular | LR/RF/GB/MLP/XGB | val balanced acc | no location (headline); location variant for comparison |
| Imagery | Timestep MLP + TemporalTransformer + CropHead | val balanced acc | masked sequences, class-balanced loss |
| CropFusion | TabTransformer + TemporalTransformer + CrossAttention + AdaptiveGatedFusion | val balanced acc | mechanism chosen on val |

## TABLE 4 — Tabular vs Imagery vs CropFusion (frozen test)
| model | balanced_accuracy | accuracy | macro_f1 | weighted_f1 | roc_auc | pepper_precision | pepper_recall |
|---|---|---|---|---|---|---|---|
| tabular | 0.4962 | 0.7571 | 0.4757 | 0.7062 | 0.4224 | 0.1818 | 0.0612 |
| imagery | 0.4847 | 0.252 | 0.2449 | 0.201 | 0.4566 | 0.1945 | 0.8724 |
| cropfusion | 0.4885 | 0.2612 | 0.2558 | 0.2175 | 0.5337 | 0.1959 | 0.8673 |

## TABLE 5 — Ablation study (validation)
| ablation | val balanced acc | val macro-F1 | val AUC |
|---|---|---|---|
| A_tabular_only | 0.5367 | 0.5159 | 0.5545 |
| A2_tabular_with_location | 0.538 | 0.4849 | 0.5689 |
| B_imagery_only | 0.5525 | 0.3771 | 0.6229 |
| C_fusion | 0.5606 | 0.3948 | 0.5408 |
| C_mechanism_concat | 0.5354 | 0.3364 | 0.5365 |
| C_mechanism_gated | 0.5567 | 0.3817 | 0.5531 |
| C_mechanism_cross | 0.5606 | 0.3948 | 0.5408 |

## TABLE 6 — Per-class precision / recall / support
| model | class | precision | recall | support |
|---|---|---|---|---|
| tabular | coconut | 0.7987 | 0.9311 | 784 |
| tabular | pepper | 0.1818 | 0.0612 | 196 |
| imagery | coconut | 0.7525 | 0.0969 | 784 |
| imagery | pepper | 0.1945 | 0.8724 | 196 |
| cropfusion | coconut | 0.7679 | 0.1097 | 784 |
| cropfusion | pepper | 0.1959 | 0.8673 | 196 |

## TABLE 7 — Robustness / confidence intervals
See `final_confidence_intervals.csv` (stratified bootstrap 95% CI, seed 42) and `final_robustness.json` (min-observation rule >= 3).
