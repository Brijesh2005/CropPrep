"""Assorted helpers shared across the CropFusion platforms.

Everything here depends only on the standard library, third-party packages
(yaml, numpy/pandas) and :mod:`shared.enums` / :mod:`shared.constants` — never
on ``training`` or ``application`` — so it can be used from either platform
without introducing cross-platform imports or circular dependencies.
"""

from __future__ import annotations

from .env import env_map_of, get_env, parse_bool, parse_int, parse_json
from .hash import md5_file, sha256_file
from .io import (
    count_lines_fast,
    copy_file_with_progress,
    human_size,
    is_csv_path,
    is_geotiff_bytes,
    is_geotiff_path,
    safe_float,
    safe_int,
    tree_signature,
    walk_files,
)
from .json_util import json_default, json_dumps, json_loads, read_json, write_json
from .naming import (
    classify_index_type,
    classify_index_type_from_path,
    classify_resolution,
    classify_resolution_from_path,
    extract_year_from_path,
    parse_observation_date,
)
from .parallel import run_parallel
from .path import ensure_dir, env_bool, is_relative_to, iter_unique_by_name_size, relposix, resolve_path
from .time import now_iso, parse_iso, to_iso, utc_now
from .yaml import dump_yaml, env_config_path, load_yaml, write_yaml, yaml_safe

__all__ = [
    "classify_index_type",
    "classify_index_type_from_path",
    "classify_resolution",
    "classify_resolution_from_path",
    "copy_file_with_progress",
    "count_lines_fast",
    "dump_yaml",
    "env_bool",
    "env_config_path",
    "env_map_of",
    "ensure_dir",
    "extract_year_from_path",
    "get_env",
    "human_size",
    "is_csv_path",
    "is_geotiff_bytes",
    "is_geotiff_path",
    "is_relative_to",
    "iter_unique_by_name_size",
    "json_default",
    "json_dumps",
    "json_loads",
    "load_yaml",
    "md5_file",
    "now_iso",
    "parse_bool",
    "parse_iso",
    "parse_int",
    "parse_json",
    "parse_observation_date",
    "read_json",
    "relposix",
    "resolve_path",
    "run_parallel",
    "safe_float",
    "safe_int",
    "sha256_file",
    "to_iso",
    "tree_signature",
    "utc_now",
    "walk_files",
    "write_json",
    "write_yaml",
    "yaml_safe",
]
