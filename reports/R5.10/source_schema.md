# R5.10 Source & Schema Audit

- **Crop_Extent** unit status: `UNKNOWN` (no authoritative parser; used ONLY as the within-field RELATIVE extent score that built the R5.9 dominant-crop target).
- **Field identity**: (taluk||hobli||village).upper() + `|SURVEYID=` + survey_id.upper() (R5.8 rule).
- **Split**: grouped field splits by taluk, inherited unchanged from R5.9.

## Survey files (govt crop survey)
- `ogd_bantvala_kharif_2020_21.csv` (21 cols, sha256 84959ba0a4d5)
- `ogd_beltangadi_kharif_2020_21.csv` (21 cols, sha256 5e10b10cf386)
- `ogd_kokkada_kharif_2020_21.csv` (21 cols, sha256 aafdc5629ddf)
- `ogd_mangaluru_a_kharif_2020_2021.csv` (21 cols, sha256 133aa2696389)
- `ogd_mangaluru_b_kharif_2020_21.csv` (21 cols, sha256 e7d7bebba155)
- `ogd_mulki_kharif_2021_22.csv` (21 cols, sha256 8a549ba37024)
- `ogd_panemangaluru_rabi_2021_2022.csv` (21 cols, sha256 5b06a15c2c8e)
- `ogd_panja_kharif_2020_21.csv` (21 cols, sha256 8b2dad7ab72d)
- `ogd_putturu_kharif_2020_21.csv` (21 cols, sha256 44a72c5fc845)
- `ogd_sulya_kharif_2020_21.csv` (21 cols, sha256 8e22ffd816e3)
- `ogd_suratkal_kharif_2020_21.csv` (21 cols, sha256 d7211e781c86)
- `ogd_uppinangadi_kharif_2021_22.csv` (21 cols, sha256 7b6a4d68635c)
- `ogd_venuru_kharif_2020_21.csv` (21 cols, sha256 11c0fe5f7517)
- `ogd_venuru_rabi_2021_22.csv` (21 cols, sha256 a13f41c1b6c6)
- `ogd_vitla_kharif_2020_21.csv` (21 cols, sha256 c17eddc7c134)

Total raw survey rows (approx): 837,069

## R5.9 field dataset (target + split + static env)
- `field_dataset_split.csv` sha256 2ef42be509e38d37

## DK grid temporal composites (2018-2023)
- `DK_Features_2018.csv` (37 cols if True)
- `DK_Features_2019.csv` (37 cols if True)
- `DK_Features_2020.csv` (37 cols if True)
- `DK_Features_2021.csv` (37 cols if True)
- `DK_Features_2022.csv` (37 cols if True)
- `DK_Features_2023.csv` (37 cols if True)

## Temporal signal source
Each field's real survey GPS is matched to the per-year DK grid with `SpatialTabularMatcher` (K-NN IDW, radius 5 km, k=5); `Yield_Proxy_NPP` is excluded by the matcher contract. The 2024 grid file is not matcher-discoverable, so temporal slots use 2018-2023.
