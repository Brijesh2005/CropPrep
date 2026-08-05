"""Canonical constants shared across the CropFusion platforms.

Centralising these values removes the duplicated literals that previously
lived independently in each package (directory names, extensions, CRS,
provider names, environment prefixes, ...).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Package / framework identity
# --------------------------------------------------------------------------- #

#: Name of the shared framework (used for logger roots, doc references).
FRAMEWORK_NAME = "cropfusion"

#: Version of the shared framework.
FRAMEWORK_VERSION = "0.1.0"

#: Default Kaggle dataset handle for the primary image dataset.
DEFAULT_KAGGLE_HANDLE = (
    "shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada"
)

#: Default catalog (dataset) name used for registry + raw layout.
DEFAULT_CATALOG_NAME = "kaggle-crop-yield"

# --------------------------------------------------------------------------- #
# Directory layout
# --------------------------------------------------------------------------- #

#: Directory name holding the canonical (materialised) copy of downloads.
DIR_RAW = "raw"

#: Directory name for derived / processed datasets.
DIR_PROCESSED = "processed"

#: Directory name for internal state (sqlite stores, scan cache).
DIR_STATE = ".cropfusion"

#: Directory name for the tabular CSVs.
DIR_TABULAR = "tabular"

#: Cache sub-directory under the state root.
DIR_CACHE = "cache"

#: Directories always skipped during discovery (internal state / VCS).
EXCLUDE_DIRS: frozenset[str] = frozenset(
    {DIR_STATE, ".cache", ".git", "__pycache__", ".idea", ".venv", "venv"}
)

# --------------------------------------------------------------------------- #
# File extensions
# --------------------------------------------------------------------------- #

#: Raster extensions treated as GeoTIFF candidates.
RASTER_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff"})

#: Extensions treated as CSV-like.
CSV_SUFFIXES: frozenset[str] = frozenset({".csv", ".txt"})

#: Config file extensions.
CONFIG_SUFFIXES: frozenset[str] = frozenset({".yaml", ".yml", ".json"})

#: Model artifact extensions.
MODEL_EXTENSIONS: frozenset[str] = frozenset({".pt", ".pth", ".ckpt", ".onnx"})

#: Vector boundary extensions.
VECTOR_SUFFIXES: frozenset[str] = frozenset({".shp", ".geojson", ".json", ".gpkg"})

# --------------------------------------------------------------------------- #
# Coordinate reference systems
# --------------------------------------------------------------------------- #

#: Sentinel-2 tile CRS used for the Karnataka study region (UTM zone 43N).
CRS_UTM_43N = "EPSG:32643"

#: Web-mercator CRS used for map tiles.
CRS_WEB_MERCATOR = "EPSG:3857"

#: Geographic CRS.
CRS_WGS84 = "EPSG:4326"

#: Common local CRS alias used by the raster metadata loader.
CRS_UNKNOWN = "unknown"

# --------------------------------------------------------------------------- #
# Image / raster formats
# --------------------------------------------------------------------------- #

#: GDAL driver name for GeoTIFF.
GDAL_DRIVER_GTIFF = "GTiff"

#: GDAL compression names considered safe defaults.
GDAL_COMPRESSION_OPTIONS: tuple[str, ...] = ("deflate", "lzw", "zstd", "none")

#: TIFF magic bytes (little endian "II*\0" and big endian "MM\0*").
TIFF_MAGIC: tuple[bytes, bytes] = (b"II*\x00", b"MM\x00*")

#: TIFF value type sizes indexed by TIFF type tag.
TIFF_TYPE_SIZE: dict[int, int] = {
    1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
    8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4,
}

# --------------------------------------------------------------------------- #
# Tabular / CSV formats
# --------------------------------------------------------------------------- #

#: Default read chunk for streaming CSV loads.
CSV_DEFAULT_CHUNK_ROWS: int = 100_000

#: Number of rows sampled to infer CSV dtypes.
CSV_DTYPE_SAMPLE_ROWS: int = 1_000

#: Number of rows returned by preview operations.
CSV_PREVIEW_ROWS: int = 5

# --------------------------------------------------------------------------- #
# Environment variable prefixes (platform + shared)
# --------------------------------------------------------------------------- #

#: Dataset Manager prefix.
ENV_PREFIX_DATASET = "DM_"

#: STAM prefix.
ENV_PREFIX_STAM = "ST_"

#: Training runner prefix.
ENV_PREFIX_TRAINING = "TD_"

#: Preprocessing prefix.
ENV_PREFIX_PREPROCESSING = "PPT_"

#: Models prefix.
ENV_PREFIX_MODELS = "MOD_"

#: Explainability prefix.
ENV_PREFIX_EXPLAINABILITY = "EXP_"

#: MLOps prefix.
ENV_PREFIX_MLOPS = "ML_"

#: Application (backend) prefix.
ENV_PREFIX_BACKEND = "BACKEND_"

#: Shared framework prefix (used when shared tooling is configured).
ENV_PREFIX_SHARED = "CF_"

#: Conventional env var pointing at a config file.
ENV_CONFIG_FILE = "CONFIG_FILE"

# --------------------------------------------------------------------------- #
# Provider names
# --------------------------------------------------------------------------- #

#: Provider for the Git-versioned tabular CSVs.
PROVIDER_GIT_TABULAR = "git-tabular"

#: Provider for the Kaggle imagery catalog.
PROVIDER_KAGGLE_IMAGE = "kaggle-image"

#: Provider for the GeoJSON boundary store.
PROVIDER_GEOJSON_BOUNDARY = "geojson-boundary"

# --------------------------------------------------------------------------- #
# Metadata keys
# --------------------------------------------------------------------------- #

#: Metadata record keys shared across platforms.
META_PATH = "path"
META_RELATIVE_PATH = "relative_path"
META_CATEGORY = "category"
META_INDEX_TYPE = "index_type"
META_RESOLUTION = "resolution"
META_YEAR = "year"
META_OBSERVATION_DATE = "observation_date"
META_CRS = "crs"
META_SHA256 = "sha256"
META_CREATED_AT = "created_at"
META_EXTRA = "extra"
META_VERSION = "version"

# --------------------------------------------------------------------------- #
# Misc sizes
# --------------------------------------------------------------------------- #

#: Streaming read chunk (1 MiB) used by hashing / copying / counting.
CHUNK_SIZE: int = 1 << 20
