# R5.9 Source & Schema Audit

- **Crop_Extent**: `A-B-C` compound string. Although A∈[0,4266], B∈[0,99], C∈[0,666] are observed, **no authoritative parser or unit documentation exists** in this repository. Therefore this phase declares `CROP_EXTENT_UNIT_STATUS = UNKNOWN`.
- **Scalarization (relative-only)**: score = A + B/100 + C/10000. This monotonic reading is used ONLY to compare crops within one field (fractions + dominant crop). No m2 / acre claim is made.
- **Field identity**: (taluk||hobli||village)+survey_id (R5.8 rule).
- **Composition exclusions**: NA Land, Fallow, Trees and Grooves, Harvest over Crop-* are never counted in the composition.

## Survey files
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

Total raw survey rows (approx): 837069