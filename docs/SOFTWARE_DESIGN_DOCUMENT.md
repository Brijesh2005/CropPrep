# CropFusion: Software Design Document

**Version:** 1.0  
**Date:** August 1, 2026  
**Status:** Draft - Architecture Phase  
**Project Type:** AI-Powered Agricultural Decision Support System  

---

## Document Control

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-08-01 | Architecture Team | Initial architecture design |

---

## Table of Contents

1. Executive Summary
2. Project Overview
3. System Architecture Overview
4. Architectural Modules
5. AI Architecture Design
6. Frontend Architecture
7. Backend Architecture
8. Database Architecture
9. Data Flow Architecture
10. API Design
11. Security Architecture
12. Deployment Architecture
13. Folder Structure
14. Development Standards
15. Technology Stack Justification
16. Functional Requirements
17. Non-Functional Requirements
18. Risk Analysis
19. Future Extensions
20. Development Roadmap
21. Milestones and Timeline

---

## 1. Executive Summary

### 1.1 Project Vision

CropFusion is a next-generation AI-powered Agricultural Decision Support System designed to revolutionize crop recommendation and yield prediction through intelligent fusion of structured agricultural data and multi-temporal satellite imagery.

Unlike traditional systems requiring manual soil parameter entry, CropFusion delivers intelligent predictions using only GPS coordinates or map selection.

### 1.2 Research Innovation

The system implements the research titled:

**"CropFusion: A Hybrid Machine Learning–Deep Learning Spatio-Temporal Cross-Modal Fusion Framework for Location-Based Multi-Crop Recommendation and Yield Prediction Using Structured Agricultural Data and Multi-Temporal Sentinel-2 Vegetation Indices"**

This represents a significant advancement in precision agriculture through:

- **Spatio-Temporal Alignment Module (STAM)**: Novel location-time-data fusion
- **Cross-Modal Architecture**: Hybrid TabTransformer + Dual CNN + Temporal Transformer
- **Multi-Task Learning**: Simultaneous crop recommendation and yield prediction
- **Explainable AI**: SHAP and attention-based interpretability

### 1.3 Core Value Proposition

**For Farmers:**
- Zero-effort predictions (GPS only)
- Crop recommendations tailored to location
- Expected yield forecasting
- AI-powered explanations
- Historical tracking

**For Researchers:**
- Novel cross-modal fusion architecture
- Reproducible research platform
- Extensible AI framework
- Comprehensive evaluation metrics

**For Agricultural Policy:**
- Data-driven insights
- Regional analysis capabilities
- Production forecasting
- Resource optimization

### 1.4 Key Differentiators

1. **Location-First Design**: No manual feature entry required
2. **Multi-Modal Fusion**: Structured data + satellite imagery
3. **Temporal Intelligence**: Seasonal sequence learning
4. **Production-Grade**: Enterprise architecture from day one
5. **Research-Grade**: Publication-ready implementation
6. **Explainable**: Transparent AI decision-making
7. **Extensible**: Designed for future enhancements

---

## 2. Project Overview

### 2.1 Problem Statement

Current agricultural recommendation systems suffer from:

1. **Manual Data Entry Burden**: Farmers must measure and input soil parameters
2. **Single-Modal Limitations**: Systems use either tabular data OR imagery, not both
3. **Temporal Blindness**: Ignoring seasonal and historical patterns
4. **Black-Box Predictions**: Lack of explainability
5. **Geographic Constraints**: Limited spatial intelligence
6. **Scalability Issues**: Not designed for production deployment

### 2.2 Solution Overview

CropFusion addresses these challenges through:

**Automated Data Retrieval**
- GPS → Nearest dataset location
- Automatic feature extraction
- Historical data integration
- Multi-temporal satellite imagery

**Multi-Modal AI**
- TabTransformer for structured data
- Dual CNN for NDVI/EVI imagery
- Temporal Transformer for sequences
- Cross-modal attention fusion

**Explainable Intelligence**
- SHAP values for tabular features
- Attention weight visualization
- GradCAM for image interpretation
- Feature importance ranking

**Production Architecture**
- Microservices backend
- React frontend
- PostGIS spatial database
- Docker deployment
- CI/CD pipeline

### 2.3 Target Users

**Primary Users**
- Farmers (smallholder to commercial)
- Agricultural extension officers
- Government agricultural departments

**Secondary Users**
- Agricultural researchers
- Data scientists
- Policy makers
- Agribusiness companies

### 2.4 Geographic Scope

**Phase 1 (MVP):**
- District: Dakshina Kannada, Karnataka, India
- Years: 2018-2025
- Villages: Dataset coverage areas

**Phase 2 (Expansion):**
- Karnataka state-wide
- Additional districts
- Extended temporal coverage

**Phase 3 (Scale):**
- Pan-India deployment
- International adaptation
- Real-time satellite integration

### 2.5 Dataset Overview

**Structured Data Sources:**
- Crop production history
- Soil characteristics (NPK, pH, organic matter)
- Weather data (rainfall, temperature, humidity)
- Yield records
- Fertilizer usage
- Seasonal patterns
- Administrative boundaries

**Satellite Imagery:**
- Source: Sentinel-2 via Kaggle dataset
- Years: 2018-2025
- Indices: NDVI, EVI
- Resolution: 10m (NDVI), 20m (EVI)
- Temporal: Multiple observations per season

### 2.6 Success Criteria

**Technical Metrics:**
- Crop recommendation accuracy >85%
- Yield prediction MAE <15%
- Response time <2 seconds
- System uptime >99.5%
- API availability >99.9%

**User Experience:**
- Zero manual feature entry
- <3 clicks to prediction
- Mobile responsive
- <5 second page load
- Intuitive map interaction

**Research Metrics:**
- Reproducible results
- Publishable architecture
- Documented methodology
- Open-source potential
- Academic citations

---

## 3. System Architecture Overview

### 3.1 High-Level Architecture


### 3.2 System Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CROPFUSION SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────┐          ┌──────────────────────────────┐            │
│   │    FRONTEND      │          │      API GATEWAY             │            │
│   │   (React SPA)    │◄────────►│  (FastAPI / Auth / Routing)  │            │
│   │  Map + Dashboard │   HTTPS  └─────────────┬────────────────┘            │
│   └──────────────────┘                        │                             │
│                    ┌─────────────────────────┼───────────────────┐         │
│                    │                         │                   │         │
│              ┌─────▼─────┐            ┌──────▼──────┐    ┌──────▼──────┐    │
│              │ INFERENCE │            │  DATASET    │    │   GIS       │    │
│              │ SERVICE   │            │  SERVICE    │    │   SERVICE   │    │
│              └─────┬─────┘            └──────┬──────┘    └──────┬──────┘    │
│                    │                         │                   │         │
│              ┌─────▼─────┐            ┌──────▼──────┐    ┌──────▼──────┐    │
│              │   STAM    │            │    DATA     │    │   Spatial   │    │
│              │  ALIGNER  │            │ INTEGRATION │    │   Index     │    │
│              └─────┬─────┘            └──────┬──────┘    └──────┬──────┘    │
│                    │                         │                   │         │
│              ┌─────▼─────┐            ┌──────▼──────┐            │         │
│              │  AI MODEL │            │    MASTER   │            │         │
│              │  ENSEMBLE │            │  AGRICULT-  │            │         │
│              │           │            │  URAL DS    │            │         │
│              └─────┬─────┘            └──────┬──────┘            │         │
│                    │                         │                   │         │
│              ┌─────▼─────┐            ┌──────▼──────┐    ┌──────▼──────┐    │
│              │ EXPLAIN-  │            │  POSTGRES   │    │   MINIO/    │    │
│              │ ABILITY   │            │  + POSTGIS  │    │   S3 STORE  │    │
│              │ SERVICE   │            │  + REDIS    │    │  (GeoTIFF)  │    │
│              └───────────┘            └─────────────┘    └─────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Architectural Principles

1. **Microservices Architecture** — independent deployable services, loose coupling, high cohesion, service-specific scaling, fault isolation, technology flexibility.

2. **Event-Driven Data Flow** — async processing for heavy tasks, message queue for dataset processing, pub/sub for system events, backpressure handling.

3. **Layered Architecture** — presentation layer (React), API layer (FastAPI Gateway), service layer (business logic), data layer (PostgreSQL, Redis, Object Storage).

4. **Clean Architecture** — dependency inversion, interface segregation, testable components, domain-driven design.

5. **Twelve-Factor App** — config in environment, stateless processes, backing services as attached resources, dev/prod parity, logging as event streams.

### 3.4 Architecture Decision Records (Summary)

| ADR | Decision | Rationale |
|-----|----------|-----------|
| ADR-001 | Microservices over Monolith | Independent scaling, team autonomy |
| ADR-002 | FastAPI over Flask/Django | Async, Pydantic, OpenAPI |
| ADR-003 | React + TypeScript over Vue | Ecosystem, type safety |
| ADR-004 | PostgreSQL + PostGIS | Spatial support, reliability |
| ADR-005 | PyTorch over TensorFlow | Research flexibility, ecosystem |
| ADR-006 | Docker Compose for dev | Reproducibility |
| ADR-007 | Leaflet over MapLibre | Simplicity for MVP |
| ADR-008 | JWT auth | Stateless, scalable |

### 3.5 System Quality Attributes

**Performance** — P95 inference latency < 2s; P95 API latency < 300ms; support 100 concurrent users; image serving < 500ms.

**Scalability** — horizontal scaling via Docker/K8s; Redis caching for hot paths; async processing for training; sharded dataset storage.

**Reliability** — health checks on all services; circuit breakers on AI calls; retry policies for external calls; database replication.

**Security** — JWT with 15-min expiry; RBAC for admin functions; rate limiting per user; input validation via Pydantic.

**Maintainability** — clear module boundaries; automated testing (80%+ coverage); documentation as code; standardized logging.

**Observability** — structured JSON logs; Prometheus metrics; Grafana dashboards; OpenTelemetry tracing.

---

## 4. Architectural Modules

### 4.1 Dataset Manager

**Purpose:** Centralized management of all datasets with automated download, validation, versioning, and indexing.

#### 4.1.1 Module Responsibilities

1. **Kaggle Dataset Download** — automated download via kagglehub for `shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada`; progress tracking; resume support; checksum verification.

2. **Dataset Validation** — file structure verification; GeoTIFF integrity checks; CSV schema validation; data quality scoring; temporal completeness checks; spatial coverage checks.

3. **Metadata Generation** — dataset statistics; file manifests; schema definitions; provenance tracking; timestamps and hashes.

4. **Dataset Versioning** — semantic versioning (MAJOR.MINOR.PATCH); snapshot on changes; version rollback support; change logs; backward compatibility.

5. **Dataset Indexing** — file path registry; spatial index (grid cells); temporal index (year/season); quick lookup tables; cache invalidation.

6. **Dataset Cache** — LRU caching of hot files; memory-mapped GeoTIFF access; prefetch optimization; disk quota management.

#### 4.1.2 Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATASET MANAGER                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ DOWNLOADER   │  │ VALIDATOR    │  │ METADATA GENERATOR│ │
│  │ kagglehub    │  │ Schema Checks│  │ Stats + Manifests │ │
│  │ resume + hash│  │ Quality Score│  │ Provenance        │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                   │            │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌────────▼─────────┐  │
│  │ VERSIONER    │  │ INDEXER      │  │ CACHE MANAGER    │  │
│  │ semver + roll│  │ spatial + temp│  │ LRU + mmap       │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         └─────────────────┼───────────────────┘            │
│                           ▼                                │
│              ┌────────────────────────┐                    │
│              │  DATASET STORAGE LAYER │                    │
│              │  Raw / Processed /     │                    │
│              │  Metadata / Cache      │                    │
│              └────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

#### 4.1.3 Dataset Storage Schema

```
datasets/
├── raw/                                    # Original downloads
│   └── kaggle-crop-yield/
│       ├── 2018_images/
│       │   ├── R10m/  ├── R20m/  ├── NDVI/  └── EVI/
│       ├── ... (2019-2025)
│       └── csv/                            # Tabular datasets
├── processed/                              # Cleaned/derived
│   ├── master_agricultural.parquet
│   ├── ndvi_sequences.parquet
│   ├── evi_sequences.parquet
│   └── feature_store/
├── metadata/                               # Version manifests
│   ├── manifests/  ├── schemas/  └── quality_reports/
└── cache/                                  # Hot files
    └── geotiff_mmap/
```

#### 4.1.4 Validation Pipeline

```
Download
   │
   ▼
SHA-256 Checksum  ──── FAIL ──► Re-download
   │
   ▼
Schema Validation  ──── FAIL ──► Report + Fix
   │
   ▼
GeoTIFF Integrity ──── FAIL ──► Remove + Report
   │
   ▼
Temporal Completeness (8 years × seasons)
   │
   ▼
Spatial Coverage (villages/coordinates)
   │
   ▼
Quality Score (0-100)
   │
   ▼
Metadata Generation + Versioning
```

#### 4.1.5 Key Design Decisions

- **kagglehub over manual SAFE archives**: automated, versioned, reproducible.
- **Parquet for tabular**: columnar, compressed, fast.
- **COG (Cloud Optimized GeoTIFF)**: chunked reads, spatial queries.
- **Immutable raw data**: never modify originals.
- **Lazy loading**: process on demand.

---

### 4.2 Data Integration Module

**Purpose:** Merge all CSV datasets into a unified Master Agricultural Dataset with cleaning, validation, and normalization.

#### 4.2.1 Module Responsibilities

1. **CSV Ingestion** — read all source CSVs; schema inference; encoding detection; type coercion; header standardization.
2. **Data Merging** — key-based joins (village, district, year, season); foreign key resolution; conflict resolution; hierarchical merge strategy; surrogate key generation.
3. **Data Cleaning** — missing value handling; outlier detection; duplicate removal; format normalization; unit standardization.
4. **Data Validation** — range checks (pH 0-14, temperature -10 to 50°C); cross-field validation; referential integrity; statistical profiling.
5. **Normalization** — min-max scaling; z-score standardization; one-hot encoding; label encoding; target encoding (with validation).
6. **Master Dataset Creation** — unified schema; consistent keys; denormalized for ML; optimized storage (Parquet); versioned snapshots.

#### 4.2.2 Merge Strategy

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  CROP.CSV    │   │  YIELD.CSV   │   │  RAINFALL    │
│  crop_id     │   │  crop_id     │   │  region      │
│  name        │   │  yield_kg   │   │  monsoon_mm  │
│  season      │   │  area_ha    │   │  annual_mm   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
              ┌─────────────────────┐
              │   MERGE ENGINE      │
              │  Join Keys:         │
              │  village_id         │
              │  district_id        │
              │  year, season       │
              │  crop_type          │
              │  Strategy:          │
              │  full outer join    │
              └──────────┬──────────┘
                         ▼
        ┌───────────────────────────────┐
        │     MASTER AGRICULTURAL DS    │
        │  (wide, denormalized table)   │
        └───────────────────────────────┘
```

#### 4.2.3 Master Dataset Schema (Target)

| Column Group | Columns | Source |
|--------------|---------|--------|
| Identifiers | sample_id (PK), village_id, district_id | System |
| Geographic | latitude, longitude, elevation_m | GIS CSV |
| Temporal | year, season (Kharif/Rabi/Zaid) | All |
| Crop | crop_type, crop_variety, planting_date | Crop CSV |
| Soil | soil_type, ph, nitrogen_ppm, phosphorus_ppm, potassium_ppm, organic_carbon_pct, soil_moisture_pct | Soil CSV |
| Weather | annual_rainfall_mm, monsoon_rainfall_mm, avg_temp_c, max_temp_c, min_temp_c, humidity_pct, sunshine_hours | Weather CSV |
| Production | area_ha, production_tons, yield_kg_per_ha | Production CSV |
| Fertilizer | fertilizer_type, fertilizer_amount_kg | Fertilizer CSV |
| Historical | prev_year_yield, 5yr_avg_yield, trend_slope | Historical CSV |
| Derived | ndvi_peak, evi_peak, growing_days | Feature Engine |

#### 4.2.4 Data Quality Rules

| Rule | Valid Range | Action |
|------|-------------|--------|
| Soil pH | 0 - 14 | Clip/flag |
| Temperature | -10 to 50°C | Flag |
| Rainfall | 0 - 5000 mm | Flag |
| Yield | 0 - 50000 kg/ha | Flag |
| Humidity | 0 - 100% | Clip |
| NDVI | -1 to 1 | Flag |
| EVI | -1 to 1 | Flag |
| Area | > 0 | Reject |

#### 4.2.5 Missing Value Strategy

| Column | Strategy | Rationale |
|--------|----------|-----------|
| Soil NPK | KNN imputation (k=5) | Correlated with region |
| Weather | Seasonal mean | Temporal stationarity |
| Yield | Drop row if target | Cannot fabricate target |
| Humidity | Linear interpolation | Temporal series |
| pH | Median by soil type | Domain knowledge |

---

### 4.3 Spatial Temporal Alignment Module (STAM)

**Purpose:** The core research contribution that aligns GPS location, time (year/season), and multimodal data into unified Agricultural Observation Samples.

#### 4.3.1 Module Overview

STAM answers: **"Given any GPS point, what is the complete agricultural context at that location for a given year and season?"**

It transforms raw spatial coordinates into a fully-featured ML-ready sample by aligning:
- Nearest dataset location
- NDVI time sequence
- EVI time sequence
- Tabular agricultural features

#### 4.3.2 STAM Pipeline (Detailed)

```
Step 1: Location Resolution
┌───────────────────────────────────────────────────────────────┐
│ Input: GPS (lat, lon)  OR  Map Click (lat, lon)               │
│                                                               │
│ Spatial Index Query (PostGIS / R-tree / Geohash Grid):        │
│   SELECT * FROM villages                                      │
│   ORDER BY geom <-> ST_MakePoint(lon, lat)                    │
│   LIMIT 1;                                                    │
│                                                               │
│ Output: nearest_village_id, distance_m, admin boundaries      │
│                                                               │
│ Threshold: if distance > max_radius (config, e.g. 5 km)       │
│   → flag "low_confidence_location"                            │
└───────────────────────────────────────────────────────────────┘

Step 2: Temporal Context
┌───────────────────────────────────────────────────────────────┐
│ Input: year (default = latest available), season               │
│                                                               │
│ Crop Calendar Lookup:                                         │
│   • Kharif:  June - October   (monsoon crops)                 │
│   • Rabi:    Nov  - March     (winter crops)                  │
│   • Zaid:    April - June     (summer crops)                  │
│                                                               │
│ Determine growing window [planting, harvest]                  │
│ Output: year, season, growing_window                          │
└───────────────────────────────────────────────────────────────┘

Step 3: Image Sequence Retrieval
┌───────────────────────────────────────────────────────────────┐
│ For each observation date in growing window:                  │
│   1. Lookup NDVI GeoTIFF for (village, year, date)            │
│   2. Extract pixel at village coordinates                     │
│   3. Apply cloud mask + quality filter                        │
│   4. Compute zonal statistics:                                │
│      - pixel_value (center)                                   │
│      - 3x3 neighborhood mean                                 │
│      - 3x3 neighborhood std                                 │
│   5. Store as [date, ndvi_value]                              │
│                                                               │
│ Repeat for EVI.                                               │
│                                                               │
│ Output:                                                       │
│   ndvi_series: [(t0,v0), (t1,v1), ..., (tn,vn)]              │
│   evi_series:  [(t0,v0), (t1,v1), ..., (tn,vn)]              │
│   dates:       [t0, t1, ..., tn]                              │
│                                                               │
│ Handling: missing dates → interpolation + flag                │
│   Out of bounds → zero-pad + mask                             │
└───────────────────────────────────────────────────────────────┘

Step 4: Tabular Retrieval
┌───────────────────────────────────────────────────────────────┐
│ From Master Agricultural Dataset:                             │
│   SELECT * FROM master_dataset                                │
│   WHERE village_id = nearest_village_id                       │
│     AND year = target_year                                    │
│     AND season = target_season                                │
│                                                               │
│ Output: tabular_features (soil, weather, production, etc.)    │
└───────────────────────────────────────────────────────────────┘

Step 5: Sample Assembly
┌───────────────────────────────────────────────────────────────┐
│                     AGRICULTURAL OBSERVATION                  │
│                              SAMPLE                           │
│ ┌───────────────┬────────────────────────────────────────┐    │
│ │ Component     │ Content                               │    │
│ ├───────────────┼────────────────────────────────────────┤    │
│ │ context       │ village_id, coords, distance, year,    │    │
│ │               │ season, admin boundaries               │    │
│ │ ndvi_sequence │ [T, 1] normalized NDVI series          │    │
│ │ evi_sequence  │ [T, 1] normalized EVI series           │    │
│ │ tabular       │ soil+weather+production feature vector │    │
│ │ masks         │ validity mask for missing timesteps    │    │
│ │ metadata      │ source files, versions, provenance     │    │
│ └───────────────┴────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
```

#### 4.3.3 STAM Research Contributions

**Contribution 1: Location-Adaptive Alignment** — automatic nearest-location resolution; distance-weighted confidence; grid-based spatial indexing (geohash + R-tree hybrid).

**Contribution 2: Temporal Sequence Synthesis** — growing-window-aware sampling; cloud-gap interpolation; season-aware augmentation.

**Contribution 3: Cross-Modal Co-Registration** — pixel-level image-tabular alignment; single coordinate reference frame; unified sample format.

**Contribution 4: Uncertainty Propagation** — location confidence; cloud coverage flags; data completeness score; all propagated to final model confidence.

#### 4.3.4 STAM Complexity Analysis

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Nearest location | O(log n) | R-tree spatial index |
| Image lookup | O(1) | Geohash-based directory |
| Pixel extraction | O(p) | p = number of points |
| Sequence assembly | O(T) | T = timesteps |
| Full pipeline | O(log n + T·p) | Real-time capable |

#### 4.3.5 STAM Configuration

```
stam:
  spatial:
    max_nearest_distance_km: 5.0
    index_cell_size_km: 0.1
    neighborhood_kernel: 3
  temporal:
    default_year: latest
    sequence_length: 8
    window_before_planting: 10   # days
    window_after_harvest: 5      # days
  images:
    cloud_threshold: 0.3
    interp_method: linear
    out_of_bounds: zero_pad
  sample:
    include_masks: true
    include_metadata: true
```

#### 4.3.6 STAM Failure Modes & Handling

| Failure | Detection | Handling |
|---------|-----------|----------|
| No location in range | Distance > threshold | Return nearest + low confidence |
| Missing images for year | File not found | Fallback to previous year + flag |
| All timesteps cloudy | Cloud mask coverage | Interpolate + flag "image_degraded" |
| Missing tabular data | No rows matched | Fallback to district aggregate |
| Partial year coverage | Date range check | Trim sequence + update mask |

---

### 4.4 Feature Engineering Module

**Purpose:** Transform raw data into model-ready features with proper train/validation/test splits.

#### 4.4.1 Module Responsibilities

1. **Tabular Preprocessing** — categorical encoding (crop, season, soil type); numerical scaling (weather, NPK); feature interactions; aggregation features.
2. **Image Preprocessing** — cloud masking; gap interpolation; normalization (NDVI/EVI ranges); fixed sequence length padding; augmentation (noise, shift, scale).
3. **Seasonal Sequence Generation** — resample to fixed timesteps; phenological features (peak, timing of peak); growth rate features; integral metrics (time-integrated NDVI).
4. **Dataset Splitting** — temporal split (no leakage); spatial holdout (village-level); stratified by crop; seed reproducibility.

#### 4.4.2 Feature Categories

**Tabular Features (TabTransformer input):**
- Categorical: crop_type, season, soil_type, district, village
- Numerical: rainfall, temperature, humidity, NPK, pH, organic_carbon, area
- Derived: 5yr_avg_yield, weather_delta, soil_fertility_index

**Image Features (CNN input):**
- NDVI sequence: [T=8, H=3, W=3] patches
- EVI sequence: [T=8, H=3, W=3] patches
- Time-integrated NDVI, green-up rate

**Sequence Features (Temporal Transformer input):**
- NDVI trajectory embedding; EVI trajectory embedding; temporal masks.

#### 4.4.3 Data Splitting Strategy

```
┌──────────────────────────────────────────────────────────┐
│                     DATASET (2018-2025)                 │
│                                                          │
│  Temporal Split (no leakage):                            │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐               │
│  │  TRAIN   │ │  VALID   │ │   TEST     │               │
│  │ 2018-2022│ │ 2023     │ │  2024-2025 │               │
│  │ (70%)    │ │ (15%)    │ │  (15%)     │               │
│  └──────────┘ └──────────┘ └────────────┘               │
│                                                          │
│  Spatial Holdout:                                        │
│  20% of villages EXCLUDED from train                     │
│  → tests generalization to unseen locations              │
│                                                          │
│  Stratification: crop_type + season balance              │
│  Reproducibility: fixed seed (42), hash-splitting        │
└──────────────────────────────────────────────────────────┘
```

**Leakage Prevention:** no train-time statistics on test set; village-level grouping; temporal block splitting; feature selection only on train.

#### 4.4.4 Augmentation Strategy

| Modality | Augmentation | When |
|----------|--------------|------|
| Image | Random noise (σ=0.01) | Train only |
| Image | Temporal shift (±1 timestep) | Train only |
| Image | Intensity scale (0.95-1.05) | Train only |
| Tabular | SMOTE for rare crops | Train only |
| Sequence | Mask augmentation (dropout 10%) | Train only |

#### 4.4.5 Feature Store Output

```
feature_store/
├── train/
│   ├── tabular.parquet
│   ├── ndvi.npy          [N, T, H, W]
│   ├── evi.npy           [N, T, H, W]
│   ├── targets.npy       [N]  (crop labels)
│   └── yields.npy        [N]  (yield targets)
├── valid/
├── test/
├── column_metadata.json
└── preprocessing.pkl    # fitted transformers
```

---

### 4.5 AI Module

**Purpose:** The core multimodal deep learning architecture for crop recommendation and yield prediction.

#### 4.5.1 Module Responsibilities

1. **Model Architecture** — TabTransformer + Dual CNN + Temporal Transformer + Cross-Modal Attention.
2. **Training Pipeline** — automatic training, checkpointing, early stopping.
3. **Evaluation** — multi-task metrics, confusion analysis.
4. **Inference** — fast, batched, production-ready.
5. **Model Registry** — versioned models, experiment tracking.
6. **Retraining** — scheduled + on-demand.

#### 4.5.2 Training Pipeline

```
Data Loader (Dataloader with prefetch)
   │
   ▼
Forward Pass
   ├── Tabular → TabTransformer → Tabular Embedding
   ├── NDVI 3D patches → CNN → Feature Map
   ├── EVI 3D patches → CNN → Feature Map
   └── Concatenate → Temporal Transformer → Image Embedding
   │
   ▼
Cross-Modal Attention
   │
   ▼
Shared Fusion Encoder
   │
   ▼
Multi-Task Heads
   ├── Crop Recommendation (classification)
   └── Yield Prediction (regression)
   │
   ▼
Loss Computation
   ├── CE_loss (classification)
   ├── SmoothL1_loss (regression)
   └── Total = α·CE + β·SmoothL1 + λ·L2_reg
   │
   ▼
Backward Pass + Optimizer Step (AdamW, warmup cosine)
   │
   ▼
Validation + Early Stopping + Checkpoint
```

#### 4.5.3 Model Outputs

**Crop Recommendation:** `{top_1_crop, probability, top_5_crops}`
**Yield Prediction:** `{predicted_yield_kg_per_ha, confidence_interval}`
**Confidence Score:** `{overall_confidence: 0-100, factors: [...]}`

---

### 4.6 Explainability Module

**Purpose:** Make AI predictions transparent, interpretable, and trustworthy.

#### 4.6.1 Module Responsibilities

1. **SHAP Analysis** — tabular feature contributions.
2. **Attention Visualization** — cross-modal attention weights.
3. **GradCAM** — image region importance.
4. **Feature Importance** — global and local.
5. **Prediction Reasoning** — natural language explanation.
6. **Confidence Calibration** — well-calibrated probabilities.

#### 4.6.2 Explanation Types

| Type | Technique | Output |
|------|-----------|--------|
| Local Tabular | SHAP (KernelSHAP) | Feature contribution bar chart |
| Global Tabular | SHAP (TreeSHAP) | Aggregate importance |
| Temporal | Attention weights | Which dates mattered most |
| Spatial | GradCAM | Which image regions mattered |
| Cross-modal | Cross-attention | Which modality drove decision |

#### 4.6.3 Explanation Generation Flow

```
Prediction Output
   │
   ▼
SHAP: Tabular Contributions
   ├── Soil factors (pH, NPK) → contribution
   ├── Weather factors → contribution
   └── History → contribution
   │
   ▼
Attention Analysis
   ├── Temporal attention → date importance
   └── Cross-modal attention → modality weights
   │
   ▼
GradCAM (on CNN feature maps) → image region heatmap
   │
   ▼
Reasoning Engine
   ├── Template-based natural language
   ├── Rule extraction from SHAP
   └── Confidence aggregation
   │
   ▼
Explanation Payload (JSON)
   ├── feature_contributions[]
   ├── temporal_importance[]
   ├── spatial_heatmap_url
   ├── modality_weights{}
   ├── reasoning_text
   └── confidence_factors[]
```

#### 4.6.4 Sample Explanation Payload

```json
{
  "reasoning": "Based on sandy loam soil (pH 6.2, moderate nitrogen),
    adequate monsoon rainfall (2200mm), strong vegetative vigor
    (peak NDVI 0.85 in August), and historical success of Rice in
    Dakshina Kannada, the model recommends Rice with 91% confidence.",
  "feature_contributions": {
    "rainfall": {"value": 2200, "contribution": 0.18},
    "soil_type": {"value": "sandy_loam", "contribution": 0.12},
    "history": {"value": "5yr_avg_yield", "contribution": 0.09},
    "temperature": {"value": 28.5, "contribution": -0.03}
  },
  "temporal_importance": {"july": 0.31, "august": 0.38, "september": 0.22},
  "modality_weights": {"tabular": 0.42, "ndvi": 0.33, "evi": 0.25}
}
```

---

### 4.7 GIS Module

**Purpose:** Interactive mapping, location selection, and spatial intelligence.

#### 4.7.1 Module Responsibilities

1. **Interactive Map Rendering** — Leaflet/MapLibre with OSM base.
2. **Location Selection** — click-to-select, GPS integration.
3. **Nearest Dataset Search** — spatial queries via PostGIS.
4. **Administrative Boundaries** — village/district/state polygons.
5. **Selectable Locations** — only dataset-covered areas.
6. **Spatial Analytics** — future (zones, heatmaps, coverage).

#### 4.7.2 Map Interaction Flow

```
User Opens Map
   │
   ▼
Load Dataset Coverage Layer (GeoJSON from GIS Service)
   │
   ▼
Two Paths:
   ├── GPS: "Use My Location" button → browser geolocation
   │        → snap to nearest dataset point
   └── Click: click map → nearest dataset point highlighted
   │
   ▼
Select Point → Confirmation Dialog
   ├── Village: "Moodabidri"
   ├── District: "Dakshina Kannada"
   └── Distance from click: 120m
   │
   ▼
"Get Prediction" → Inference Service
```

#### 4.7.3 GIS Data Model

```
┌────────────────────────┐   ┌────────────────────────┐
│      VILLAGE           │   │      DISTRICT          │
│  id: UUID              │   │  id: UUID              │
│  name: varchar         │   │  name: varchar         │
│  geom: Point (PostGIS) │   │  geom: MultiPolygon    │
│  district_id: FK       │   │  state_id: FK          │
│  coverage: boolean     │   └───────────┬────────────┘
└──────────┬─────────────┘               │
           │  1:N                        │
           ▼                             ▼
┌────────────────────────┐   ┌────────────────────────┐
│   DATASET_LOCATION     │   │     STATE              │
│  village_id: FK        │   │  id: UUID              │
│  lat/lon: float        │   │  name: varchar         │
│  has_imagery: boolean  │   │  geom: MultiPolygon    │
│  has_tabular: boolean  │   └────────────────────────┘
│  years_covered: int[]  │
│  seasons: varchar[]    │
└────────────────────────┘
```

#### 4.7.4 Spatial Indexing

- GiST index on geometry columns.
- Geohash grid for fast lookups.
- Precomputed KNN for performance.
- Tiling for map rendering.

---

### 4.8 Backend Services

**Purpose:** FastAPI microservices providing REST APIs for the frontend.

#### 4.8.1 Service Decomposition

```
┌─────────────────────────────────────────────────────────────┐
│                      API GATEWAY (FastAPI)                 │
│  /api/v1/*   Auth middleware, rate limiting, routing       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ AUTH     │ │ INFERENCE│ │ DATASET  │ │  GIS         │  │
│  │ SERVICE  │ │ SERVICE  │ │ SERVICE  │ │  SERVICE     │  │
│  │ JWT, RBAC│ │ STAM+AI  │ │ Manager  │ │  Spatial     │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ HISTORY  │ │ EXPLAIN- │ │ ADMIN    │ │  MONITORING  │  │
│  │ SERVICE  │ │ ABILITY  │ │ SERVICE  │ │  SERVICE     │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 4.8.2 Service Responsibilities

| Service | Responsibilities |
|---------|-----------------|
| API Gateway | Routing, auth, rate limiting, CORS, aggregation |
| Auth Service | Registration, login, JWT, refresh tokens, RBAC |
| Inference Service | STAM alignment, model inference, prediction API |
| Dataset Service | Dataset status, metadata, version info, management |
| GIS Service | Coverage layers, nearest location, boundaries |
| History Service | Prediction history CRUD, user timelines |
| Explainability Service | SHAP, attention, GradCAM computation |
| Admin Service | User management, system metrics, dataset ops |
| Monitoring | Health, metrics, logs aggregation |

#### 4.8.3 Inter-Service Communication

- **Synchronous**: HTTP/REST via gateway.
- **Async**: Redis Pub/Sub, Celery queues.
- **Events**: Prediction events, dataset events.
- **Circuit breaking**: resilience patterns.

---

### 4.9 Frontend Module

**Purpose:** Professional React application for farmers and admins.

#### 4.9.1 Page Architecture

```
Frontend (React + TypeScript + Tailwind)
├── Landing Page
│   ├── Hero + "Start Prediction" CTA, Features, How it works
├── Prediction Page
│   ├── GPS button, Interactive Map (Leaflet)
│   ├── Selected Location Panel, "Get Prediction" button
├── Prediction Dashboard
│   ├── Recommended Crop card, Yield Prediction card
│   ├── Confidence gauge, Comparison chart (top-5 crops)
│   └── Historical trends
├── Explainability View
│   ├── Feature contribution chart, Temporal importance timeline
│   ├── Spatial heatmap overlay, AI reasoning text
├── History Page
│   ├── List of past predictions, Filters, Detail drill-down
├── Profile Page
│   ├── User info, Farm locations, Preferences
├── Settings Page
│   ├── Theme (dark/light), Language, Notification prefs
├── Admin Dashboard
│   ├── User management, System metrics, Model performance
└── Dataset Management Dashboard
    ├── Dataset status, Version history
    ├── Download triggers, Quality metrics
```

#### 4.9.2 Component Architecture

```
components/
├── layout/            # App shell, nav, sidebar
├── map/               # Leaflet wrapper, location picker
├── prediction/        # Prediction cards, gauges
├── explainability/    # Charts, heatmaps, reasoning
├── history/           # History list, filters
├── admin/             # Admin tables, metrics
├── common/            # Button, Card, Modal, Table, Form
├── charts/            # Recharts wrappers
└── auth/              # Login, register, guards
```

#### 4.9.3 State Management

- **Server State**: TanStack Query (React Query).
- **Client State**: Zustand.
- **Form State**: React Hook Form.
- **Route State**: React Router v6.
- **Theme**: Tailwind dark mode class strategy.

---

### 4.10 Database Module

**Purpose:** Reliable, scalable, spatial-aware data persistence.

#### 4.10.1 Database Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      POSTGRESQL + POSTGIS                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  APPLICATION SCHEMA (OLTP)                          │    │
│  │  users, predictions, datasets, model_versions      │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  GIS SCHEMA (Spatial)                               │    │
│  │  villages, districts, boundaries, coverage         │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  ANALYTICS SCHEMA (OLAP)                            │    │
│  │  prediction_stats, model_metrics, usage_metrics    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  REDIS (Cache + Queue + Session)                            │
│  • Prediction result cache (1hr TTL)                        │
│  • Rate limit counters, Celery broker/backend               │
│  • Feature lookup cache (STAM)                              │
│                                                             │
│  OBJECT STORAGE (MinIO/S3)                                  │
│  • Raw GeoTIFFs, Processed patches                          │
│  • Model artifacts (weights), Explainability heatmaps       │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. AI Architecture

### 5.1 Complete Neural Architecture

#### 5.1.1 Architecture Diagram

```
                        CROPFUSION MULTIMODAL AI ARCHITECTURE
                        ═══════════════════════════════════════

   TABULAR BRANCH                              IMAGE BRANCH
 ┌───────────────────┐                 ┌───────────────────────────────────────┐
 │ Categorical Feats │                 │    NDVI SEQUENCE [T, H, W]            │
 │ (crop, season,    │                 │    EVI SEQUENCE  [T, H, W]            │
 │  soil, district)  │                 │                                       │
 └─────────┬─────────┘                 └──────────────┬────────────────────────┘
           │                                          │
 ┌─────────▼─────────┐                 ┌──────────────▼────────────────────────┐
 │   EMBEDDINGS      │                 │        DUAL 3D-CNN ENCODER            │
 │   categorical     │                 │                                       │
 │   (learned)       │                 │   ┌──────────────┐  ┌──────────────┐  │
 └─────────┬─────────┘                 │   │  NDVI CNN    │  │  EVI CNN     │  │
           │                          │   │  Conv3D-BN   │  │  Conv3D-BN   │  │
 ┌─────────▼─────────┐                 │   │  ReLU-Pool   │  │  ReLU-Pool   │  │
 │ NUMERICAL FEATS   │                 │   │  ...         │  │  ...         │  │
 │ (rainfall, NPK,   │                 │   │  Flatten     │  │  Flatten     │  │
 │  temp, humidity)  │                 │   └──────┬───────┘  └──────┬───────┘  │
 └─────────┬─────────┘                 └─────────┼──────────────────┼─────────┘
           │                                     │                  │
 ┌─────────▼─────────┐                           └────────┬─────────┘
 │  TABTRANSFORMER   │                                    │
 │  (Nx Transformer  │                           ┌────────▼─────────┐
 │   Encoder Blocks) │                           │  IMAGE FUSION   │
 │                   │                           │  (concat + proj) │
 └─────────┬─────────┘                           └────────┬─────────┘
           │                                     ┌────────▼─────────┐
           │                                     │  TEMPORAL        │
           │                                     │  TRANSFORMER     │
           │                                     │  (cross-time      │
           │                                     │   attention)      │
           │                                     └────────┬─────────┘
           │                                     ┌────────▼─────────┐
           │                                     │  IMAGE           │
           │                                     │  EMBEDDING       │
           │                                     │  (CLS token)     │
           │                                     └────────┬─────────┘
           │                                              │
 ┌─────────▼──────────────────────────────────────────────▼─────────┐
 │                    CROSS-MODAL ATTENTION                         │
 │   Q = Tabular Embedding   K,V = Image Embedding                 │
 │   (tabular attends to images)  +  (images attend to tabular)    │
 │   → Cross-modal fusion with learned weights                      │
 └──────────────────────────────────┬──────────────────────────────┘
                                    │
 ┌──────────────────────────────────▼──────────────────────────────┐
 │                     SHARED FUSION ENCODER                       │
 │        (2x Transformer Blocks + LayerNorm + Dropout)            │
 └──────────────────────────────────┬──────────────────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
 ┌──────────────▼─────────────┐         ┌───────────────▼─────────────┐
 │   CROP RECOMMENDATION HEAD │         │   YIELD PREDICTION HEAD     │
 │   (Softmax over N crops)   │         │   (Regression head)         │
 │   Loss: CrossEntropy       │         │   Loss: Smooth L1 / Huber   │
 │   Output: top-1 + probs    │         │   Output: yield kg/ha       │
 └────────────────────────────┘         └─────────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────┐
                       │  UNCERTAINTY HEAD     │
                       │  (auxiliary, future)  │
                       │  MC-Dropout variance  │
                       └──────────────────────┘
```

#### 5.1.2 Detailed Layer Specifications

**Tabular Branch — TabTransformer**

```
Input:
  categorical_features: [C_cat]   (one-hot / label indexed)
  numerical_features: [C_num]

Layer 1: Embedding
  categorical → Learned Embedding, dim=64
  numerical  → Linear(dim→64) + LayerNorm
  → token sequence [C_cat + 1, 64]  (with CLS token)

Layer 2-N: Transformer Encoder
  N=4 blocks
  Each: MultiHeadAttention(4 heads, dim=64)
        + FeedForward(64→256→64, GELU)
        + Pre-LayerNorm + Residual + Dropout(0.1)

Output:
  Tabular Embedding [1, 64]   (CLS token pooled)
```

**Image Branch — Dual CNN + Temporal Transformer**

```
Input:
  NDVI patches: [T=8, H=3, W=3, 1]
  EVI patches:  [T=8, H=3, W=3, 1]

NDVI CNN:
  Conv3D(1→32, k=3, s=1) + BN + ReLU + Pool
  Conv3D(32→64, k=3) + BN + ReLU + Pool
  Flatten → Linear → [128]

EVI CNN: (identical structure, separate weights) → [128]

Image Fusion:
  concat([ndvi_vec, evi_vec]) → Linear(256→128) → [128]
  (positional encoding added for temporal order)

Temporal Transformer:
  N=2 blocks, MultiHeadAttention(4 heads)
  → [T, 128] → CLS pooled → Image Embedding [128]
```

**Cross-Modal Attention**

```
CrossAttention:
  Q_tab = Wq · tabular_embedding        [1, 64]
  K_img = Wk · image_embedding          [1, 128]
  V_img = Wv · image_embedding          [1, 128]
  attn  = softmax(Q·Kᵀ/√d)              [1, 1]
  fused = concat([tabular, attn·V])     [192]

Symmetrically: images attend to tabular
  fused_full = concat([tab→img, img→tab])  [192]
```

**Multi-Task Heads**

```
Shared Fusion Encoder: [192] → 2× Transformer block → [192]

Head 1 — Crop Recommendation:
  Linear(192→128) + ReLU + Dropout
  Linear(128→N_crops) + Softmax
  Loss: CrossEntropy (weight = 0.7)

Head 2 — Yield Prediction:
  Linear(192→128) + ReLU + Dropout
  Linear(128→1)  (relu clamp ≥ 0)
  Loss: SmoothL1(β=0.5) (weight = 0.3)

Total Loss = 0.7·CE + 0.3·SmoothL1 + 1e-4·L2
```

#### 5.1.3 Why This Architecture Beats CNN + XGBoost

| Dimension | CNN + XGBoost | CropFusion Hybrid |
|-----------|---------------|-------------------|
| Modalities | Handled separately (pipeline) | Unified end-to-end |
| Temporal | Manual feature extraction | Learned temporal attention |
| Spatial | Fixed pre-computed indices | Learned spatial CNN features |
| Cross-modal | No interaction | Cross-modal attention |
| Representation | Task-specific shallow | Shared deep representations |
| Explainability | SHAP only (tabular) | SHAP + attention + GradCAM |
| Multi-task | Separate models | Joint multi-task training |
| Uncertainty | None | MC-dropout, confidence calibration |
| End-to-end | Not differentiable | Fully differentiable |
| Feature learning | Manual feature engineering | Automatic hierarchical learning |

**Technical Rationale:**

1. **End-to-End Gradient Flow** — the fusion encoder learns representations that jointly optimize both tasks, capturing cross-modal interactions a pipeline approach cannot.

2. **Cross-Modal Attention > Late Concatenation** — explicitly learns which modality matters for each prediction, adapting per-location. XGBoost cannot model interaction between image sequences and tabular features.

3. **Temporal Transformer > Hand-crafted Phenology Metrics** — learns optimal temporal attention (when in the season matters) instead of relying on pre-defined indices.

4. **Multi-Task Learning** — shared representation + task-specific heads improves both tasks via regularization and positive transfer.

5. **Uncertainty Estimation** — confidence calibration is research-grade; XGBoost provides no principled confidence.

6. **Extensibility** — adding disease detection or health heads is a plug-in; XGBoost requires a new pipeline.

#### 5.1.4 Training Configuration

```
training:
  batch_size: 32
  epochs: 100 (early stopping patience 10)
  optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
  scheduler: Cosine annealing with warmup (5 epochs)
  loss_weights: {crop: 0.7, yield: 0.3}
  augmentation: enabled
  precision: mixed (fp16)
  device: cuda (single or DDP)
  seed: 42
```

#### 5.1.5 Model Registry & Versioning

```
model_registry/
├── v1.0.0/
│   ├── model.pt            # weights
│   ├── config.yaml         # hyperparameters
│   ├── metrics.json        # eval metrics
│   ├── preprocessing.pkl   # fitted transformers
│   └── metadata.json       # data version, git hash, env
├── v1.1.0/
└── latest/                 # symlink to current
```

---

## 6. Frontend Architecture

### 6.1 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Framework | React 18 | Ecosystem, performance |
| Language | TypeScript 5 | Type safety |
| Build | Vite | Fast HMR, modern |
| Styling | Tailwind CSS 3 | Utility-first, dark mode |
| Maps | Leaflet + react-leaflet | Lightweight, mature |
| Charts | Recharts | Composable, SVG |
| Data Fetching | TanStack Query | Caching, invalidation |
| State | Zustand | Minimal, scalable |
| Routing | React Router 6 | Standard |
| Forms | React Hook Form | Performant |
| Validation | Zod | Runtime safety |
| Testing | Vitest + Testing Library | Fast, integrated |

### 6.2 Routing Structure

```
/                    → Landing
/predict             → Prediction (map + GPS)
/predict/result      → Prediction Dashboard
/explain/:id         → Explainability View
/history             → History
/profile             → Profile
/settings            → Settings
/login               → Auth
/register            → Auth
/admin               → Admin Dashboard
/admin/datasets      → Dataset Management
```

### 6.3 Component Design Principles

- **Atomic Design**: atoms → molecules → organisms → templates.
- **Server State**: all API data via TanStack Query.
- **Suspense**: lazy loading routes and heavy components.
- **Accessibility**: WCAG 2.1 AA compliance.
- **Responsive**: mobile-first, breakpoints sm/md/lg/xl.
- **Dark Mode**: class strategy with system preference.
- **Design Tokens**: centralized theme in `theme.ts`.

### 6.4 Data Fetching Pattern

```
User Action (click GPS / map)
   │
   ▼
react-query: usePrediction(lat, lon)
   ├── mutation/query to /api/v1/predictions
   ├── loading state → skeleton components
   ├── success → store result in Zustand
   └── error → friendly retry UI
```

### 6.5 Prediction Dashboard UI Design

```
┌──────────────────────────────────────────────────────────────┐
│  CROPFUSION  ·  Prediction Dashboard                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │ RECOMMENDED     │  │ YIELD           │  │ CONFIDENCE   │  │
│  │ CROP            │  │ PREDICTION      │  │ GAUGE        │  │
│  │  🌾 Rice        │  │  5,200 kg/ha    │  │   ╭─────╮   │  │
│  │  91%            │  │  CI [4,900-5,500]│  │  ╰──87──╯  │  │
│  └─────────────────┘  └─────────────────┘  └──────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ TOP-5 CROP COMPARISON (bar chart)                      │  │
│  │  Rice ████████████████████████ 0.91                     │  │
│  │  Arecanut ██████████ 0.42                               │  │
│  │  Coconut ████████ 0.31                                  │  │
│  │  Banana ██████ 0.24                                     │  │
│  │  Pepper █████ 0.19                                      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ LOCATION: Moodabidri, Dakshina Kannada · dist 120 m    │  │
│  │ SEASON: Kharif 2025   ·  [View Explanation →]          │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Backend Architecture

### 7.1 FastAPI Service Template (per service)

```
┌───────────────────────────────────────────────────────────┐
│                    SERVICE TEMPLATE                      │
├───────────────────────────────────────────────────────────┤
│  app/                                                    │
│  ├── main.py            # FastAPI app, middleware         │
│  ├── api/               # routers (versioned)            │
│  ├── core/              # config, security, deps         │
│  ├── models/            # SQLAlchemy models              │
│  ├── schemas/           # Pydantic schemas               │
│  ├── services/          # business logic                 │
│  ├── repositories/      # data access layer              │
│  ├── workers/           # Celery tasks                   │
│  ├── utils/             # helpers                         │
│  └── tests/             # unit + integration             │
└───────────────────────────────────────────────────────────┘
```

### 7.2 Inference Service Request Flow

```
POST /api/v1/predictions
  │
  ▼
Body: {lat, lon, year?, season?}
  │
  ▼
Validate (Pydantic): range checks, required fields
  │
  ▼
Check Cache (Redis): key = sha256(lat,lon,year,season)
  ├── HIT → return cached prediction
  └── MISS
       ▼
STAM Align: resolve location → sequences + tabular
       │
       ▼
Model Inference: run multi-task model
       │
       ▼
Explainability (async): enqueue explanation job
       │
       ▼
Save History: insert into predictions table
       │
       ▼
Cache Store: write result (TTL 1hr)
       │
       ▼
Return Response
```

### 7.3 Celery Task Queue

| Task | Queue | Purpose |
|------|-------|---------|
| train_model | training | Model training job |
| explain_prediction | explain | SHAP/attention computation |
| build_features | etl | Feature store build |
| validate_dataset | dataset | Dataset validation |
| generate_metadata | dataset | Metadata generation |
| nightly_metrics | metrics | Usage/metric aggregation |

### 7.4 Error Handling Strategy

- Global exception handler.
- Structured error responses `{code, message, details, trace_id}`.
- Retry with exponential backoff.
- Circuit breakers for external calls.
- Graceful degradation (fallback models).

---

## 8. Database Architecture

### 8.1 Entity Relationship Diagram

```
┌──────────────────┐       ┌──────────────────────┐
│      USERS       │       │    PREDICTIONS        │
├──────────────────┤       ├──────────────────────┤
│ id: UUID (PK)    │ 1    N│ id: UUID (PK)         │
│ email: varchar   │◄──────│ user_id: FK           │
│ password_hash    │       │ location_id: FK       │
│ full_name        │       │ model_version_id: FK  │
│ role: enum       │       │ lat: float            │
│ is_active: bool  │       │ lon: float            │
│ created_at       │       │ crop_prediction: json │
│ updated_at       │       │ yield_prediction: json│
└──────────────────┘       │ confidence: float     │
                           │ status: enum          │
                           │ created_at            │
                           └──────────┬────────────┘
                                      │
              ┌───────────────────────┼──────────────┐
              │                       │              │
 ┌────────────▼─────────┐  ┌──────────▼────────┐  ┌──▼───────────────────────┐
 │   LOCATIONS          │  │ EXPLANATIONS      │  │  MODEL_VERSIONS          │
 ├──────────────────────┤  ├───────────────────┤  ├──────────────────────────┤
 │ id: UUID (PK)        │  │ id: UUID (PK)     │  │ id: UUID (PK)            │
 │ village_id: FK       │  │ prediction_id: FK │  │ version: varchar         │
 │ lat: float           │  │ shap_json: json   │  │ artifact_path: varchar   │
 │ lon: float           │  │ attention_json    │  │ metrics_json             │
 │ name: varchar        │  │ gradcam_url       │  │ status: enum             │
 │ is_dataset: bool     │  │ reasoning_text    │  │ created_at               │
 │ created_at           │  │ created_at        │  │                          │
 └──────────────────────┘  └───────────────────┘  └──────────────────────────┘

 ┌──────────────────────┐  ┌───────────────────┐
 │      VILLAGES        │  │     DATASETS      │
 ├──────────────────────┤  ├───────────────────┤
 │ id: UUID (PK)        │  │ id: UUID (PK)     │
 │ name: varchar        │  │ name: varchar     │
 │ district_id: FK      │  │ source: enum      │
 │ geom: geometry(POINT)│  │ version: varchar  │
 │ elevation_m: float   │  │ status: enum      │
 │ created_at           │  │ checksum: varchar │
 └──────────────────────┘  │ metadata_json     │
                           │ created_at        │
 ┌──────────────────────┐  └───────────────────┘
 │      DISTRICTS       │
 ├──────────────────────┤  ┌───────────────────┐
 │ id: UUID (PK)        │  │  REFRESH_TOKENS   │
 │ name: varchar        │  ├───────────────────┤
 │ state_id: FK         │  │ id: UUID (PK)     │
 │ geom: MultiPolygon   │  │ user_id: FK       │
 │ created_at           │  │ token_hash        │
 └──────────────────────┘  │ expires_at        │
                           │ revoked: bool     │
 ┌──────────────────────┐  └───────────────────┘
 │        STATES        │
 ├──────────────────────┤  ┌───────────────────┐
 │ id: UUID (PK)        │  │   RATE_LIMITS     │
 │ name: varchar        │  ├───────────────────┤
 │ geom: MultiPolygon   │  │ id: UUID (PK)     │
 └──────────────────────┘  │ user_id: FK       │
                           │ endpoint: varchar │
 ┌──────────────────────┐  │ window_start      │
 │  CROP_RECOMMENDATIONS│  │ request_count     │
 ├──────────────────────┤  └───────────────────┘
 │ id: UUID (PK)        │
 │ village_id: FK       │  ┌───────────────────┐
 │ year: int            │  │ SYSTEM_METRICS    │
 │ season: enum         │  ├───────────────────┤
 │ crop: varchar        │  │ id: UUID (PK)     │
 │ probability: float   │  │ metric_name       │
 │ yield_pred: float    │  │ value: float      │
 │ created_at           │  │ timestamp         │
 └──────────────────────┘  └───────────────────┘
```

### 8.2 Indexing Strategy

| Table | Index | Type |
|-------|-------|------|
| locations | (lat, lon) | BTREE composite |
| villages | geom | GiST (spatial) |
| districts | geom | GiST (spatial) |
| predictions | user_id, created_at | BTREE composite |
| predictions | location_id | BTREE |
| predictions | status | BTREE |
| explanations | prediction_id | UNIQUE |
| model_versions | version | UNIQUE |
| datasets | name, version | UNIQUE composite |
| refresh_tokens | token_hash | UNIQUE |

### 8.3 Query Patterns

**Nearest Location Query:**
```sql
SELECT v.id, v.name, v.geom,
       ST_Distance(v.geom, ST_MakePoint(:lon, :lat)) AS dist
FROM villages v
ORDER BY v.geom <-> ST_MakePoint(:lon, :lat)
LIMIT 1;
```

**Recent Predictions:**
```sql
SELECT * FROM predictions
WHERE user_id = :uid
ORDER BY created_at DESC
LIMIT 20;
```

### 8.4 Redis Key Design

```
cache:pred:{sha}          → prediction JSON (TTL 3600s)
cache:loc:{geohash}       → location resolution (TTL 86400s)
cache:feat:{key}          → STAM features (TTL 86400s)
ratelimit:{user}:{ep}     → sliding window counter
celery:task:{id}          → task status
```

---

## 9. Data Flow Architecture

### 9.1 End-to-End Prediction Data Flow

```
┌────────┐   ┌──────────────────┐   ┌──────────────────┐
│  USER  │──►│  FRONTEND        │──►│  API GATEWAY     │
│ (GPS / │   │  Map/Click/Form  │   │  /api/v1/...     │
│  Click)│   └──────────────────┘   └────────┬─────────┘
└────────┘                                  │
                                            ▼
                                  ┌──────────────────┐
                                  │ INFERENCE SERVICE│
                                  └────────┬─────────┘
                                           │
                        ┌──────────────────┼──────────────────┐
                        ▼                  ▼                  ▼
                 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                 │  GIS SERVICE │  │  STAM        │  │  REDIS CACHE│
                 │  nearest loc │  │  alignment   │  │  check      │
                 └──────────────┘  └──────────────┘  └──────────────┘
                                           │
                        ┌──────────────────┼──────────────────┐
                        ▼                  ▼                  ▼
                 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                 │  DATASET     │  │  TABULAR     │  │  IMAGE       │
                 │  MANAGER     │  │  STORE       │  │  STORE       │
                 └──────────────┘  └──────────────┘  └──────────────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │  AI MODEL    │
                                   │  (multi-task)│
                                   └──────────────┘
                                           │
                        ┌──────────────────┼──────────────────┐
                        ▼                  ▼                  ▼
                 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                 │  CROP REC    │  │  YIELD PRED  │  │  CONFIDENCE  │
                 │  (top-5)     │  │  (kg/ha)     │  │  + factors   │
                 └──────────────┘  └──────────────┘  └──────────────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │ EXPLAINABIL- │  (async)
                                   │ ITY SERVICE  │──► SHAP + Attention
                                   └──────────────┘    + GradCAM + Reasoning
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │   FRONTEND   │
                                   │  Dashboard   │
                                   └──────────────┘
```

### 9.2 Training Data Flow (Offline)

```
Kaggle Dataset
   │
   ▼
Dataset Manager (download, validate, version)
   │
   ▼
CSV Datasets → Data Integration Module → Master DS
   │                                          │
   ▼                                          ▼
Image Store (GeoTIFF)              Feature Engineering
   │                                    │
   └────────────────────────────────────┘
                    │
                    ▼
              STAM Sampling
                    │
                    ▼
          Train / Valid / Test splits
                    │
                    ▼
              Model Training
                    │
                    ▼
              Model Registry
                    │
                    ▼
              Deployment (inference service)
```

### 9.3 Async Data Flow (Dataset Refresh)

```
Admin Action: "Update Dataset"
   │
   ▼
Dataset Service → Celery task
   │
   ▼
Download (kagglehub) → Validate → Generate metadata
   │
   ▼
Version bump → Master DS rebuild (Data Integration)
   │
   ▼
Feature rebuild → Model retrain trigger
   │
   ▼
Model evaluation → Deploy if improved
   │
   ▼
Notify admin via websocket/email
```

### 9.4 Sequence Diagram — Online Prediction

```
  Frontend      API Gateway      Inference      STAM        Redis     Model     DB
     │                │              │            │          │         │        │
     │  POST /pred    │              │            │          │         │        │
     │───────────────►│              │            │          │         │        │
     │                │  Auth check  │            │          │         │        │
     │                │──────────────│            │          │         │        │
     │                │  cache GET   │            │          │         │        │
     │                │──────────────│───────────►│          │         │        │
     │                │              │            │   MISS   │         │        │
     │                │              │  align     │          │         │        │
     │                │              │────────────│─────────►│         │        │
     │                │              │            │ features │         │        │
     │                │              │            │──────────│         │        │
     │                │              │  infer     │          │         │        │
     │                │              │────────────│──────────│────────►│        │
     │                │              │            │          │         │        │
     │                │              │  save hist │          │         │        │
     │                │              │────────────│──────────│─────────│───────►│
     │                │              │  cache SET │          │         │        │
     │                │              │────────────│──────────►│         │        │
     │                │              │  enqueue xai            │         │        │
     │                │              │─────────────(async)     │         │        │
     │  JSON result   │              │            │          │         │        │
     │◄───────────────│◄─────────────│            │          │         │        │
     │                │              │            │          │         │        │
```

---

## 10. API Design

### 10.1 API Conventions

```
Base URL: /api/v1
Format: JSON
Auth: Bearer token (JWT)
Versioning: URI-based (/api/v1/...)
Pagination: ?page=&page_size= (default 20, max 100)
Errors: RFC 7807 problem details
Rate Limits: 60 req/min/user (predictions: 10/min)
```

### 10.2 Authentication APIs

```
POST   /api/v1/auth/register
  Body: {email, password, full_name}
  Returns: {access_token, refresh_token, user}

POST   /api/v1/auth/login
  Body: {email, password}
  Returns: {access_token, refresh_token, user}

POST   /api/v1/auth/refresh
  Body: {refresh_token}
  Returns: {access_token}

POST   /api/v1/auth/logout
  Auth: Bearer
  Body: {refresh_token}

GET    /api/v1/users/me
  Auth: Bearer
  Returns: {user_profile}
```

### 10.3 Prediction APIs

```
POST   /api/v1/predictions
  Auth: Bearer (optional for anonymous)
  Body: {
    "lat": 12.9724,
    "lon": 75.2834,
    "year": 2025,          // optional, default latest
    "season": "Kharif"     // optional
  }
  Returns: {
    "prediction_id": "uuid",
    "status": "completed",
    "recommended_crop": {
      "crop": "Rice",
      "probability": 0.91,
      "top_5": [{"crop": "Rice", "prob": 0.91}, ...]
    },
    "yield_prediction": {
      "crop": "Rice",
      "predicted_yield_kg_per_ha": 5200,
      "confidence_interval": [4900, 5500]
    },
    "confidence": {
      "overall": 87,
      "location": 95,
      "image_quality": 82,
      "data_completeness": 90
    },
    "location": {
      "village": "Moodabidri",
      "district": "Dakshina Kannada",
      "state": "Karnataka",
      "distance_m": 120,
      "dataset": true
    }
  }

GET    /api/v1/predictions/{id}          → full prediction record
GET    /api/v1/predictions?user=me&page=1 → paginated history
DELETE /api/v1/predictions/{id}          → owner or admin
```

### 10.4 Explainability APIs

```
GET    /api/v1/predictions/{id}/explanation
  Auth: Bearer
  Returns: {
    "reasoning_text": "...",
    "feature_contributions": [...],
    "temporal_importance": {...},
    "modality_weights": {...},
    "spatial_heatmap_url": "...",
    "attention_plot_url": "..."
  }

GET    /api/v1/predictions/{id}/explanation/refresh
  Recomputes explanation (async, returns job id)
```

### 10.5 Dataset APIs

```
GET    /api/v1/datasets                → list with status/version
GET    /api/v1/datasets/{id}           → details + metadata_json
POST   /api/v1/datasets/{id}/validate  → trigger validation (admin)
POST   /api/v1/datasets/{id}/refresh   → trigger rebuild (admin)
GET    /api/v1/datasets/versions       → version history
GET    /api/v1/datasets/coverage       → GeoJSON coverage for map
```

### 10.6 GIS APIs

```
GET    /api/v1/gis/coverage            → selectable points (GeoJSON)
GET    /api/v1/gis/nearest?lat=&lon=   → nearest dataset location
GET    /api/v1/gis/villages?district=  → villages in district
GET    /api/v1/gis/boundaries          → admin boundaries
```

### 10.7 History APIs

```
GET    /api/v1/history                 → user's predictions (paginated)
  Query: ?crop=&from=&to=&location=
GET    /api/v1/history/stats           → aggregate stats
```

### 10.8 Admin APIs

```
GET    /api/v1/admin/users             → list users, roles, activity
PATCH  /api/v1/admin/users/{id}        → update role/status
GET    /api/v1/admin/metrics           → system metrics
GET    /api/v1/admin/models            → model versions, metrics
POST   /api/v1/admin/models/{v}/deploy → deploy a model version
POST   /api/v1/admin/models/train      → trigger retraining
```

### 10.9 Health & Monitoring APIs

```
GET    /health/live    → liveness probe
GET    /health/ready   → readiness probe (DB, Redis)
GET    /health/ai      → model loaded status
GET    /metrics        → Prometheus metrics
```

---

## 11. Security Architecture

### 11.1 Authentication Flow

```
┌──────────┐         ┌──────────────┐        ┌─────────────┐
│  CLIENT  │         │  API GATEWAY │        │ AUTH SERVICE│
└────┬─────┘         └──────┬───────┘        └──────┬──────┘
     │  POST /auth/login    │                       │
     │─────────────────────►│                       │
     │                      │  POST /auth/login     │
     │                      │──────────────────────►│
     │                      │                       │
     │                      │  verify (argon2id)    │
     │                      │  issue JWT pair       │
     │                      │◄──────────────────────│
     │  {access, refresh}   │                       │
     │◄─────────────────────│                       │
     │                      │                       │
     │  GET /predictions    │                       │
     │  Bearer access       │                       │
     │─────────────────────►│                       │
     │                      │  verify JWT signature │
     │                      │  check expiry + role  │
     │                      │  forward to service   │
```

### 11.2 Security Measures

| Layer | Measure |
|-------|---------|
| Transport | HTTPS/TLS 1.3 everywhere |
| Auth | JWT access (15 min) + refresh (7 days, rotating) |
| Passwords | Argon2id hashing |
| RBAC | Roles: farmer, admin, researcher |
| Rate Limiting | Sliding window per user/IP/endpoint |
| Input Validation | Pydantic schemas, strict validation |
| SQL Injection | SQLAlchemy ORM (parameterized) |
| CORS | Whitelist origins |
| Secrets | HashiCorp Vault / .env + Doppler |
| Audit | Full audit log of admin actions |
| Data Privacy | PII minimization, encryption at rest |
| Headers | CSP, HSTS, X-Frame-Options, nosniff |

### 11.3 JWT Structure

```
Header:  {alg: HS256, typ: JWT}
Payload: {sub: user_id, role: farmer,
          iat: ..., exp: ..., jti: uuid}
Signature: HMAC-SHA256(secret)
```

### 11.4 Rate Limit Policies

| Endpoint | Anonymous | Farmer | Admin |
|----------|-----------|--------|-------|
| /predictions | 5/min | 30/min | 120/min |
| /history | 10/min | 60/min | 300/min |
| /datasets | 5/min | 20/min | 60/min |
| /gis | 30/min | 120/min | 600/min |

### 11.5 Secrets Management

```
Secrets stored in Vault (prod) / .env (dev)
Categories:
  • DB credentials (PostgreSQL, Redis)
  • JWT signing keys (rotated monthly)
  • Kaggle API credentials
  • Model registry keys
  • External API keys
Rotation: monthly automated + on-rotation alerts
```

---

## 12. Deployment Architecture

### 12.1 Environment Matrix

| Environment | Purpose | URL |
|-------------|---------|-----|
| Development | Local dev (Docker Compose) | localhost:3000 |
| Staging | Pre-prod validation | staging.cropfusion.app |
| Production | Live service | cropfusion.app |

### 12.2 Docker Deployment Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCKER COMPOSE / K8S                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐    ┌───────────────┐                     │
│  │ FRONTEND      │    │ API GATEWAY   │                     │
│  │ nginx:alpine  │    │ FastAPI       │                     │
│  │ (static SPA)  │    │ uvicorn       │                     │
│  └───────────────┘    └───────┬───────┘                     │
│        ┌──────────────────────┼──────────────────┐          │
│        ▼                      ▼                  ▼          │
│  ┌────────────┐        ┌────────────┐    ┌────────────┐     │
│  │ INFERENCE  │        │ DATASET    │    │ GIS        │     │
│  │ SERVICE    │        │ SERVICE    │    │ SERVICE    │     │
│  │ + AI GPU   │        │ + Celery   │    │            │     │
│  └────────────┘        └────────────┘    └────────────┘     │
│  ┌────────────┐        ┌────────────┐    ┌────────────┐     │
│  │ AUTH       │        │ HISTORY    │    │ EXPLAIN    │     │
│  │ SERVICE    │        │ SERVICE    │    │ SERVICE    │     │
│  └────────────┘        └────────────┘    └────────────┘     │
│  ┌────────────┐        ┌────────────┐    ┌────────────┐     │
│  │ POSTGRES   │        │ REDIS      │    │ MINIO      │     │
│  │ +POSTGIS   │        │            │    │ (S3)       │     │
│  └────────────┘        └────────────┘    └────────────┘     │
│  ┌────────────┐        ┌────────────┐    ┌────────────┐     │
│  │ PGADMIN    │        │ CELERY     │    │ GRAFANA    │     │
│  │ (dev only) │        │ WORKER     │    │ + PROM     │     │
│  └────────────┘        └────────────┘    └────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 12.3 CI/CD Pipeline (GitHub Actions)

```
[Push / PR to main]
   │
   ▼
Stage 1: Lint & Format
   ├── backend: ruff, mypy
   └── frontend: eslint, prettier
   │
   ▼
Stage 2: Unit Tests
   ├── backend: pytest (coverage > 80%)
   └── frontend: vitest
   │
   ▼
Stage 3: Build
   ├── backend: docker build + push (GHCR)
   └── frontend: vite build + docker build
   │
   ▼
Stage 4: Integration Tests (docker compose up + e2e)
   │
   ▼
Stage 5: Security Scan (trivy + semgrep)
   │
   ▼
Stage 6: Deploy to Staging + smoke tests
   │
   ▼
Stage 7: Deploy to Production (manual approval gate)
```

### 12.4 Environment Management

```
.env.example (committed)
.env.development
.env.staging
.env.production

Config via pydantic-settings BaseSettings
ALL CONFIG IS ENVIRONMENT-DRIVEN (12-factor)
```

### 12.5 GPU Inference Server

- NVIDIA T4/A10 for inference.
- ONNX Runtime or TensorRT optimization.
- Model quantization (INT8 optional).
- Warm model instances.
- Auto-scaling based on queue depth.

---

## 13. Folder Structure

### 13.1 Complete Repository Structure

```
cropfusion/
├── README.md                        # Project overview & quickstart
├── CLAUDE.md                        # AI assistant guidance
├── LICENSE                          # Open source license
├── Makefile                         # Common task shortcuts
├── docker-compose.yml               # Local dev orchestration
├── docker-compose.prod.yml          # Production orchestration
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore rules
├── .editorconfig                    # Editor consistency
│
├── frontend/                        # React TypeScript SPA
│   ├── src/
│   │   ├── app/                     # App entry, routes
│   │   ├── components/              # Reusable UI components
│   │   ├── features/                # Feature modules
│   │   ├── hooks/                   # Custom React hooks
│   │   ├── services/                # API clients
│   │   ├── stores/                  # Zustand stores
│   │   ├── types/                   # TypeScript types
│   │   ├── utils/                   # Helper functions
│   │   └── styles/                  # Global styles, theme
│   ├── public/                      # Static assets
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── Dockerfile
│
├── backend/                         # FastAPI microservices
│   ├── gateway/                     # API Gateway service
│   ├── auth_service/                # Authentication service
│   ├── inference_service/           # Prediction service
│   ├── dataset_service/             # Dataset management
│   ├── gis_service/                 # Spatial service
│   ├── history_service/             # History service
│   ├── explainability_service/      # XAI service
│   └── admin_service/               # Admin service
│
├── ai/                              # ML/DL research & training
│   ├── data/
│   │   ├── dataset_manager/         # Kaggle download, validation
│   │   ├── integration/             # Data merging, master DS
│   │   ├── stam/                    # Spatial temporal alignment
│   │   └── features/                # Feature engineering
│   ├── models/
│   │   ├── tab_transformer.py       # Tabular transformer
│   │   ├── dual_cnn.py              # NDVI/EVI CNNs
│   │   ├── temporal_transformer.py  # Temporal attention
│   │   ├── cross_modal.py           # Cross-modal attention
│   │   ├── fusion_encoder.py        # Shared encoder
│   │   ├── multi_task_heads.py      # Output heads
│   │   └── cropfusion.py            # Full model
│   ├── training/
│   │   ├── trainer.py               # Training loop
│   │   ├── callbacks.py             # Checkpoint, early stop
│   │   ├── loss.py                  # Multi-task losses
│   │   ├── metrics.py               # Evaluation metrics
│   │   └── scheduler.py             # LR scheduling
│   ├── explainability/
│   │   ├── shap_explainer.py
│   │   ├── attention_viz.py
│   │   ├── grad_cam.py
│   │   ├── reasoning.py
│   │   └── calibration.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── report.py
│   ├── registry/                    # Model registry
│   │   ├── registry.py
│   │   └── artifacts/
│   ├── configs/                     # YAML configs
│   │   ├── model.yaml
│   │   ├── training.yaml
│   │   ├── data.yaml
│   │   └── explainability.yaml
│   └── requirements.txt
│
├── services/                        # Shared service code
│   ├── common/                      # Shared libraries
│   │   ├── auth/                    # JWT, RBAC middleware
│   │   ├── logging/                 # Structured logging
│   │   ├── tracing/                 # OpenTelemetry
│   │   ├── cache/                   # Redis wrapper
│   │   ├── db/                      # SQLAlchemy base
│   │   ├── storage/                 # MinIO/S3 client
│   │   ├── errors/                  # Error handling
│   │   └── models/                  # Shared DB models
│   └── prototypes/                  # Shared contracts
│
├── database/                        # Database assets
│   ├── migrations/                  # Alembic migrations
│   ├── seeds/                       # Seed data scripts
│   ├── schemas/                     # SQL schemas
│   └── indexes/                     # Index definitions
│
├── deployment/                      # Deployment assets
│   ├── docker/                      # Dockerfiles
│   ├── nginx/                       # Nginx configs
│   ├── github-actions/              # CI/CD workflows
│   ├── monitoring/                  # Prometheus, Grafana
│   └── scripts/                     # Deploy scripts
│
├── research/                        # Research artifacts
│   ├── notebooks/                   # Jupyter notebooks
│   ├── experiments/                 # Experiment records
│   ├── baselines/                   # Baseline models
│   └── papers/                      # Literature, references
│
├── docs/                            # Documentation
│   ├── SOFTWARE_DESIGN_DOCUMENT.md  # This document
│   ├── architecture/                # Diagrams
│   ├── api/                         # API documentation
│   ├── database/                    # DB design docs
│   ├── deployment/                  # Deployment guides
│   ├── research/                    # Research notes
│   └── guides/                      # User/admin guides
│
├── tests/                           # Test suites
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Cross-service tests
│   ├── e2e/                         # End-to-end tests
│   └── fixtures/                    # Test data
│
├── configs/                         # Global configs
│   ├── logging.yaml
│   ├── monitoring.yaml
│   └── feature_flags.yaml
│
├── scripts/                         # Utility scripts
│   ├── setup/
│   ├── data/
│   ├── training/
│   ├── deployment/
│   └── maintenance/
│
└── datasets/                        # Dataset storage
    ├── raw/                         # Original downloads
    ├── processed/                   # Cleaned datasets
    ├── feature_store/               # ML features
    ├── metadata/                    # Version metadata
    └── cache/                       # Hot cache
```

### 13.2 Folder Justification

| Folder | Why It Exists |
|--------|---------------|
| frontend/ | Isolated client app, independent CI/CD, separate team |
| backend/ | Microservice boundary, per-service scaling/deploy |
| ai/ | Research-train-production separation, GPU-specific builds |
| services/ | Shared code without duplication (DRY across microservices) |
| database/ | Versioned schema migrations (Alembic), DB as code |
| deployment/ | Infra as code, reproducible environments |
| research/ | Reproducible experiments, paper traceability |
| docs/ | Living documentation, onboarding, compliance |
| tests/ | Separate test lifecycle, CI integration |
| configs/ | Environment-agnostic app config, no hardcoding |
| scripts/ | Repeatable automation, dev tooling |
| datasets/ | Data locality, storage management, gitignored |

---

## 14. Development Standards

### 14.1 Python Standards

**Tooling:** Python 3.11+; Poetry or uv for dependency management; Ruff for linting + formatting; mypy for static type checking; pytest + pytest-cov for testing.

**Style (PEP 8):** 4-space indentation; 88-char line limit (Black-compatible); snake_case for functions/variables; PascalCase for classes; UPPER_SNAKE_CASE for constants; type hints on all public functions; Google-style docstrings.

**Code Quality:** type hints everywhere; dataclasses/Pydantic for data structures; dependency injection via FastAPI; no circular imports; composition over inheritance.

### 14.2 React/TypeScript Standards

**Tooling:** TypeScript strict mode; ESLint (airbnb-based) + Prettier; Vitest + React Testing Library; Vite as build tool.

**Style:** functional components + hooks only; PascalCase for components; camelCase for variables/functions; `use` prefix for hooks; props typed via interfaces; no `any` (use `unknown` + narrowing).

**Naming:** components `PredictionCard.tsx`; hooks `usePrediction.ts`; stores `usePredictionStore.ts`; types `prediction.types.ts`.

### 14.3 General Naming Conventions

| Context | Convention | Example |
|---------|-----------|---------|
| Python files | snake_case | `tabular_preprocessor.py` |
| React files | PascalCase | `PredictionDashboard.tsx` |
| DB tables | plural snake_case | `predictions`, `users` |
| DB columns | singular snake_case | `user_id`, `created_at` |
| API endpoints | kebab-case | `/api/v1/prediction-history` |
| Git branches | feature/xxx | `feature/stam-module` |
| Docker images | cropfusion/service | `cropfusion/inference` |
| Config files | lower_snake.yaml | `model.yaml` |

### 14.4 Documentation Standards

- All public functions documented (Google style).
- README per service with run instructions.
- Architecture decisions as ADRs.
- Inline comments for non-obvious logic.
- Code comments in English.
- API changes documented with examples.

### 14.5 Testing Standards

| Layer | Coverage | Standard |
|-------|----------|----------|
| Unit (AI) | 85% | Test each module in isolation |
| Unit (Backend) | 85% | Mock external deps |
| Component (FE) | 80% | Render + interactions |
| Integration | Core flows | Service-to-service |
| E2E | Critical paths | Playwright |

### 14.6 Formatting Standards

- Ruff format (Python) / Prettier (TS).
- Pre-commit hooks enforced.
- Black-compatible line length.
- Consistent import ordering.
- No trailing whitespace.
- UTF-8 encoding everywhere.

---

## 15. Technology Stack Justification

### 15.1 Backend

| Technology | Choice | Alternatives | Why Chosen |
|-----------|--------|--------------|------------|
| API Framework | FastAPI | Django, Flask, Express | Async, Pydantic validation, OpenAPI auto |
| Language | Python | Node, Go | AI ecosystem dominance |
| ORM | SQLAlchemy 2.0 | Django ORM, Prisma | Async support, raw SQL escape |
| Task Queue | Celery | RQ, Prefect | Mature, Redis-backed, retries |
| Validation | Pydantic v2 | Marshmallow | Speed, Rust core |

### 15.2 Frontend

| Technology | Choice | Alternatives | Why Chosen |
|-----------|--------|--------------|------------|
| Framework | React 18 | Vue, Svelte, Angular | Ecosystem, hiring, maturity |
| Language | TypeScript | JavaScript | Type safety at scale |
| Build | Vite | Webpack, CRA | Speed, modern defaults |
| Styling | Tailwind | CSS modules, MUI | Rapid dev, dark mode |
| Maps | Leaflet | MapLibre, Google Maps | Open-source, lightweight |
| Charts | Recharts | Chart.js, D3 | React-native, composable |
| Data Fetch | TanStack Query | SWR, RTK Query | Caching, invalidation |

### 15.3 AI/ML

| Technology | Choice | Alternatives | Why Chosen |
|-----------|--------|--------------|------------|
| DL Framework | PyTorch | TensorFlow, JAX | Research flexibility, dynamic graphs |
| Tabular | TabTransformer | FT-Transformer, XGBoost | Transformer power on categoricals |
| Explainability | SHAP | LIME, Captum | Game-theoretic rigor |
| Image | PyTorch CNN | Keras | Deep integration |
| Experiment Track | MLflow | W&B, ClearML | Self-hostable, registry |

### 15.4 Data & Infrastructure

| Technology | Choice | Alternatives | Why Chosen |
|-----------|--------|--------------|------------|
| Database | PostgreSQL 16 | MySQL, MongoDB | Reliability, JSONB, maturity |
| Spatial | PostGIS 3 | ArcGIS, GeoServer | Powerful spatial SQL |
| Cache | Redis 7 | Memcached | Structures, pub/sub, TTL |
| Object Storage | MinIO | AWS S3, Ceph | S3-compatible, self-hosted |
| Container | Docker | Podman | Universal standard |
| Orchestration | Docker Compose | K8s, ECS | Simplicity for MVP |
| Monitoring | Prometheus+Grafana | Datadog, New Relic | Open-source, powerful |

### 15.5 Justification Summary

**Why Python + FastAPI:** the entire ML ecosystem (PyTorch, SHAP, rasterio, numpy) is Python-native. FastAPI provides the fastest async Python framework with automatic OpenAPI docs — ideal for an AI-heavy backend.

**Why React + TypeScript:** largest ecosystem, best type safety for complex dashboards, strong tooling, and the most hires available.

**Why PostgreSQL + PostGIS:** the spatial capability is native and battle-tested. A single DB handles relational + spatial + JSONB — simplifying the architecture.

**Why PyTorch:** dynamic computation graphs fit research iteration. HuggingFace, rasterio, and the broader research ecosystem are PyTorch-first.

**Why Leaflet over MapLibre:** lighter, simpler, and sufficient for point-based selection in the MVP. MapLibre remains a drop-in upgrade.

---

## 16. Functional Requirements

### 16.1 FR-01: User Authentication
- FR-01.1 Register with email + password
- FR-01.2 Login/Logout
- FR-01.3 JWT token refresh
- FR-01.4 Profile management
- FR-01.5 Role-based access (farmer/admin/researcher)

### 16.2 FR-02: Location Selection
- FR-02.1 GPS geolocation button
- FR-02.2 Interactive map click selection
- FR-02.3 Only dataset-covered locations selectable
- FR-02.4 Nearest location snapping with distance display
- FR-02.5 Location detail panel (village, district)

### 16.3 FR-03: Prediction
- FR-03.1 Crop recommendation (top-1 + top-5)
- FR-03.2 Yield prediction (kg/ha + confidence interval)
- FR-03.3 Confidence score (overall + factors)
- FR-03.4 Year/season selector (optional)
- FR-03.5 Prediction history saved automatically

### 16.4 FR-04: Explainability
- FR-04.1 SHAP feature contributions
- FR-04.2 Temporal importance (dates)
- FR-04.3 Modality weights (tabular/image)
- FR-04.4 Spatial heatmap (GradCAM)
- FR-04.5 Natural language reasoning

### 16.5 FR-05: GIS Features
- FR-05.1 Interactive map rendering
- FR-05.2 Dataset coverage overlay
- FR-05.3 Administrative boundary layers
- FR-05.4 Distance calculation
- FR-05.5 Location search (village name)

### 16.6 FR-06: Dataset Management
- FR-06.1 Automatic Kaggle download
- FR-06.2 Validation + quality reporting
- FR-06.3 Versioning + metadata
- FR-06.4 Dataset status dashboard
- FR-06.5 Manual refresh trigger

### 16.7 FR-07: History & Reporting
- FR-07.1 Prediction history list
- FR-07.2 Filters (crop/date/location)
- FR-07.3 Detail view with explanation
- FR-07.4 CSV export

### 16.8 FR-08: Admin Features
- FR-08.1 User management
- FR-08.2 Model version management
- FR-08.3 System metrics dashboard
- FR-08.4 Dataset operations
- FR-08.5 Audit log viewing

---

## 17. Non-Functional Requirements

### 17.1 NFR-01: Performance
- P95 prediction API latency < 2s
- P95 general API latency < 300ms
- Page load < 3s (P75)
- 100 concurrent prediction requests
- Model inference < 1s per request

### 17.2 NFR-02: Scalability
- Horizontal scaling of stateless services
- Auto-scaling inference on queue depth
- Redis cache absorbing read spikes
- Celery workers scaling for batch jobs
- Database read replicas

### 17.3 NFR-03: Availability
- 99.5% uptime target
- Graceful degradation without model (cached)
- Circuit breakers on dependencies
- Automatic restarts on failure
- Backup + restore < 4 hours

### 17.4 NFR-04: Security
- OWASP Top 10 compliance
- JWT auth with rotation
- Argon2id password hashing
- Input validation (Pydantic)
- Rate limiting
- Secrets encrypted at rest

### 17.5 NFR-05: Maintainability
- Test coverage > 80%
- Standardized logging + tracing
- Clear module boundaries
- Documentation kept current
- CI/CD enforced quality gates

### 17.6 NFR-06: Usability
- Mobile responsive (375px+)
- Dark/light mode
- WCAG 2.1 AA
- Kannada language support (future)
- Clear error states

### 17.7 NFR-07: Reliability
- Idempotent prediction requests
- Retry with exponential backoff
- Data integrity constraints
- Deterministic model behavior (seeded)
- Full audit trail

### 17.8 NFR-08: Extensibility
- Plugin architecture for new models
- Feature flag system
- Config-driven pipeline
- Adapter pattern for data sources
- API versioning

---

## 18. Risk Analysis

### 18.1 Technical Risks

| ID | Risk | Probability | Impact | Mitigation |
|----|------|-------------|--------|------------|
| TR-01 | Dataset download failure | Medium | High | Retry, mirrors, checksums |
| TR-02 | GeoTIFF corruption | Low | High | Validation pipeline |
| TR-03 | Model overfitting | High | High | Regularization, split strategy |
| TR-04 | Cloud occlusion in images | High | Medium | Cloud masks, interpolation |
| TR-05 | Feature leakage | Medium | High | Temporal split, spatial holdout |
| TR-06 | Model drift over years | Medium | Medium | Retraining schedule, monitoring |
| TR-07 | GPU availability | Medium | Medium | Cloud GPU, queue system |
| TR-08 | Cross-modal mismatch | Medium | Medium | Alignment validation tests |

### 18.2 Data Risks

| ID | Risk | Mitigation |
|----|------|------------|
| DR-01 | Missing years/seasons | Coverage checks, fallbacks |
| DR-02 | Inconsistent units | Unit standardization |
| DR-03 | Small sample per village | Aggregation, augmentation |
| DR-04 | Label noise | Validation, human review |
| DR-05 | Kaggle dataset updates | Version pinning |

### 18.3 Operational Risks

| ID | Risk | Mitigation |
|----|------|------------|
| OR-01 | Key person dependency | Documentation, code review |
| OR-02 | Scope creep | Roadmap, milestones |
| OR-03 | Security breach | OWASP, audits, rate limiting |
| OR-04 | Data privacy concerns | PII minimization, T&C |
| OR-05 | Deployment failures | CI/CD gates, rollback |

### 18.4 Assumptions

1. Kaggle dataset remains accessible with consistent structure.
2. Sentinel-2 imagery covers all target villages.
3. GPS accuracy adequate for village-level resolution.
4. Farmers have internet access on mobile devices.
5. Dataset labels (yield, crop) are reliable.
6. 8 years (2018-2025) provides sufficient temporal diversity.
7. Village-level aggregation is acceptable granularity.
8. Model deployment can run on a single GPU initially.

---

## 19. Future Extensions

### 19.1 Extension Architecture

All extensions plug into existing adapter interfaces:

```
┌─────────────────────────────────────────────────────────────┐
│                     EXTENSION LAYERS                       │
├─────────────────────────────────────────────────────────────┤
│  E1: Google Earth Engine    ── new image source adapter    │
│  E2: Weather APIs           ── live weather feature source │
│  E3: IoT Sensors            ── real-time soil telemetry    │
│  E4: Drone Images           ── high-res multi-spectral     │
│  E5: Disease Detection      ── new multi-task head         │
│  E6: Fertilizer Rec         ── new output head             │
│  E7: Water Requirement      ── new regression head         │
│  E8: LLM Chat Assistant     ── RAG over predictions        │
│  E9: Mobile App             ── PWA / React Native          │
└─────────────────────────────────────────────────────────────┘
```

### 19.2 Extension Details

**E1 Google Earth Engine:** adapter replacing/augmenting Kaggle imagery; real-time cloud-free composites; Landsat fusion for longer history.

**E2 Weather APIs:** OpenWeatherMap / IMD data; live rainfall/temperature for current season; forecast integration for future planning.

**E3 IoT Sensors:** MQTT ingestion for soil moisture/temperature; real-time soil telemetry replacing estimates; edge preprocessing before ingestion.

**E4 Drone Images:** high-resolution canopy imagery; precision disease/health mapping; orthomosaic processing pipeline.

**E5 Disease Detection:** additional classification head (healthy/early/late); trained on labeled disease imagery; attention-based localization.

**E6 Fertilizer Recommendation:** regression head for NPK dosage; based on soil deficit analysis; cost optimization objective.

**E7 Water Requirement:** evapotranspiration (ET) prediction; irrigation scheduling integration; climatic water balance modeling.

**E8 LLM Chat Assistant:** RAG over prediction history + knowledge base; natural language queries ("Why rice?"); Kannada language support; grounded in SHAP explanations.

**E9 Mobile Application:** PWA first (works offline); React Native for native apps; push notifications for alerts.

---

## 20. Development Roadmap

### 20.1 Phase 1: Foundation (Weeks 1-6)

**Goal:** Project scaffolding, dataset pipeline, database schema.

- Repository setup, Git flow, CI skeleton
- Dataset Manager (download, validate, version)
- Data Integration (master dataset)
- PostgreSQL + PostGIS schema + migrations
- Backend service templates
- Frontend scaffold (Vite + React + Tailwind)
- Docker Compose development environment

### 20.2 Phase 2: Data & Features (Weeks 7-12)

**Goal:** STAM, feature engineering, baselines.

- STAM implementation (location, sequence, sample)
- Feature engineering pipeline
- Baseline models (XGBoost, simple CNN)
- Train/valid/test splits
- Evaluation framework
- Research notebooks (exploration)

### 20.3 Phase 3: AI Core (Weeks 13-20)

**Goal:** Full CropFusion model, training, explainability.

- TabTransformer module
- Dual CNN modules
- Temporal Transformer
- Cross-modal attention
- Multi-task heads + training loop
- Experiment tracking (MLflow)
- Explainability (SHAP, attention, GradCAM)
- Hyperparameter optimization
- Model registry + versioning

### 20.4 Phase 4: Platform (Weeks 21-28)

**Goal:** Full backend, frontend, deployment.

- Inference service + STAM integration
- Auth service + JWT + RBAC
- GIS service + coverage layers
- History + explainability services
- Frontend prediction flow
- Admin dashboard
- Dataset management dashboard
- Production deployment (Docker Compose)
- Monitoring (Prometheus + Grafana)

### 20.5 Phase 5: Hardening (Weeks 29-36)

**Goal:** Security, testing, performance, release.

- Security audit + fixes
- Load testing + optimization
- E2E test suite
- Documentation completion
- Beta user testing
- Production release v1.0.0

### 20.6 Development Team (Recommended)

| Role | Count | Responsibility |
|------|-------|----------------|
| Backend Engineer | 1 | Services, API, DB |
| ML Engineer | 1 | Models, training, XAI |
| Frontend Engineer | 1 | React app, maps |
| GIS/Data Engineer | 1 | Spatial data, ETL |
| DevOps Engineer | 0.5 | CI/CD, deployment |
| Technical Lead | 1 | Architecture, review |

---

## 21. Milestones and Timeline

### 21.1 Milestone Summary

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| M0: Architecture Approved | Week 1 | SDD signed off |
| M1: Data Pipeline Complete | Week 6 | Dataset Manager + Master DS |
| M2: STAM Complete | Week 10 | STAM + feature pipeline |
| M3: Baseline Results | Week 12 | Baseline model metrics |
| M4: CropFusion Model | Week 18 | Full model beats baselines |
| M5: Explainability | Week 20 | SHAP + attention working |
| M6: Platform Beta | Week 26 | End-to-end prediction flow |
| M7: Production Ready | Week 32 | Hardened, monitored, secured |
| M8: v1.0.0 Release | Week 36 | Public release |

### 21.2 Timeline Gantt

```
Week:      1   6   10  12  18  20  26  28  32  36
            │   │   │   │   │   │   │   │   │   │
Architecture ●───┤
Data Pipeline  ●───┤
STAM                ●───┤
Features                ●─┤
Baselines               │  ●─┤
CropFusion Model            ●───────┤
Explainability                        ●───┤
Platform                                ●───────┤
Hardening                                       ●───────┤
Release                                                    ●
```

### 21.3 Delivery Phases

**MVP (Month 1-6):** location → prediction core; single district (Dakshina Kannada); basic explainability; web app.

**v1.0 (Month 9):** full explainability suite; admin + dataset dashboards; production deployment; mobile responsive.

**v1.5 (Month 12+):** additional districts; weather API integration; LLM chat assistant; fertilizer recommendations.

---

## Appendix A: Technology Justification Matrix

### A.1 Complete Stack Summary

| Layer | Technology | Version | License |
|-------|-----------|---------|---------|
| Frontend | React | 18.x | MIT |
| Frontend | TypeScript | 5.x | Apache |
| Frontend | Vite | 5.x | MIT |
| Frontend | Tailwind CSS | 3.x | MIT |
| Frontend | Leaflet | 1.9 | BSD-2 |
| Frontend | Recharts | 2.x | MIT |
| Backend | FastAPI | 0.11x | MIT |
| Backend | Uvicorn | 0.3x | BSD-3 |
| Backend | Pydantic | 2.x | MIT |
| Backend | SQLAlchemy | 2.x | MIT |
| Backend | Celery | 5.x | BSD-3 |
| AI | PyTorch | 2.x | BSD-3 |
| AI | torchvision | 0.18 | BSD-3 |
| AI | SHAP | 0.44 | MIT |
| AI | rasterio | 1.3 | BSD-3 |
| AI | numpy | 1.26 | BSD-3 |
| AI | pandas | 2.x | BSD-3 |
| AI | scikit-learn | 1.4 | BSD-3 |
| AI | MLflow | 2.x | Apache |
| Data | PostgreSQL | 16 | PostgreSQL |
| Data | PostGIS | 3.4 | GPLv2 |
| Data | Redis | 7.x | BSD-3 |
| Data | MinIO | 2024 | AGPLv3 |
| Infra | Docker | 25 | Apache |
| Infra | Nginx | 1.25 | BSD-2 |
| Infra | Prometheus | 2.x | Apache |
| Infra | Grafana | 10.x | AGPLv3 |
| Testing | pytest | 8.x | MIT |
| Testing | Vitest | 1.x | MIT |
| Testing | Playwright | 1.x | Apache |
| CI/CD | GitHub Actions | - | - |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| STAM | Spatial Temporal Alignment Module — aligns GPS, time, and multimodal data into samples |
| NDVI | Normalized Difference Vegetation Index — vegetation greenness |
| EVI | Enhanced Vegetation Index — robust to soil/atmosphere |
| TabTransformer | Transformer architecture for tabular data |
| Cross-Modal Attention | Attention mechanism fusing multiple data modalities |
| SHAP | SHapley Additive exPlanations — feature attribution |
| GradCAM | Gradient-weighted Class Activation Mapping |
| Kharif | Monsoon cropping season (June-October) |
| Rabi | Winter cropping season (Nov-March) |
| Zaid | Summer cropping season (April-June) |
| MAE | Mean Absolute Error |
| CE | Cross-Entropy loss |
| GeoTIFF | Georeferenced raster image format |
| PostGIS | Spatial extension for PostgreSQL |
| RBAC | Role-Based Access Control |
| JWT | JSON Web Token |
| COG | Cloud Optimized GeoTIFF |

---

## Appendix C: References

1. Huang et al. — TabTransformer: Tabular Data Modeling Using Contextual Embeddings (2020)
2. Vaswani et al. — Attention Is All You Need (2017)
3. Selvaraju et al. — Grad-CAM: Visual Explanations from Deep Networks (2017)
4. Lundberg & Lee — A Unified Approach to Interpreting Model Predictions (2017)
5. Sentinel-2 Mission Documentation — ESA
6. Kaggle Dataset: shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada
7. Hu et al. — Squeeze-and-Excitation Networks (2018)
8. Hendrycks & Gimpel — Gaussian Error Linear Units (2016)
9. Loshchilov & Hutter — AdamW (2019)
10. FAO — Crop Water Requirement Guidelines

---

*This Software Design Document represents the complete architecture for the CropFusion system. It is designed to be a living document that evolves with the project.*

**END OF DOCUMENT — PHASE 1 (ARCHITECTURE) COMPLETE**
