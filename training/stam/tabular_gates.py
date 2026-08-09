"""Gate A / Gate B auto-approval review for tabular sources.

``review()`` wraps :func:`training.stam.tabular_profiler.profile_tabular_source`
with two gates that decide whether a CSV may be auto-wired into the live
``training/config/stam.yaml`` ``tabular.tables`` list:

- **Gate A (wiring sanity)** — a source is wireable end-to-end only when its
  location column is high-confidence, its year column is unambiguous, and a
  name-based yield column + crop column exist. Wide-format ICRISAT-style
  tables always fail Gate A because their crop/yield are *derived* from the
  dominant crop — a manual decision.

- **Gate B (name safety)** — every unique place name must resolve against the
  boundary/alias vocabulary. New unambiguous fuzzy aliases are **suggested**
  (``--add-aliases`` appends them to ``name_aliases.py``) but never block.
  Unmapped or ambiguous names block auto-approval: those rows would silently
  drop at match time.

CLI: ``python -m training.stam.tabular_gates``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .tabular_profiler import (
    LOCATION_CONFIDENCE_MIN,
    PlaceNameVerdict,
    SourceProfile,
    candidate_table_entry,
    clean_name,
    key_name,
    profile_tabular_source,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABULAR_DIR = _REPO_ROOT / "training" / "datasets" / "tabular"
DEFAULT_STAM_YAML = _REPO_ROOT / "training" / "config" / "stam.yaml"
DEFAULT_NAME_ALIASES = _REPO_ROOT / "training" / "stam" / "name_aliases.py"


@dataclass(frozen=True)
class GateCheck:
    """One named gate check with a pass/fail verdict and human detail."""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class TableReview:
    """The full Gate A + Gate B review of one source profile."""

    profile: SourceProfile
    gate_a: list[GateCheck]
    gate_b: list[GateCheck]
    warnings: list[str]
    suggested_aliases: list[PlaceNameVerdict]

    @property
    def approved(self) -> bool:
        return all(check.ok for check in self.gate_a + self.gate_b)

    @property
    def entry(self) -> dict:
        return candidate_table_entry(self.profile)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def gate_a(profile: SourceProfile) -> list[GateCheck]:
    """Wiring-sanity checks: location, year, yield, crop."""
    checks: list[GateCheck] = []

    location = profile.location
    if location.column and location.confidence >= LOCATION_CONFIDENCE_MIN:
        checks.append(GateCheck(
            "location", True,
            f"{location.column} @ {location.confidence:.0%} confidence ({location.note})",
        ))
    else:
        checks.append(GateCheck("location", False, "no high-confidence location column"))

    year = profile.year
    if year.column and year.method not in ("no-year-column", "ambiguous-year"):
        checks.append(GateCheck("year", True, f"{year.column} ({year.note})"))
    else:
        checks.append(GateCheck("year", False, "no unambiguous year column"))

    yield_ = profile.yield_
    if profile.wide_format:
        checks.append(GateCheck(
            "yield", False,
            "wide-format source: crop/yield derived from the dominant crop "
            "(not auto-approved)",
        ))
    elif yield_.column and yield_.method == "name-based-match":
        checks.append(GateCheck("yield", True, f"{yield_.column} (name-based match)"))
    else:
        checks.append(GateCheck("yield", False, "no name-based yield column"))

    crop = profile.crop
    if profile.wide_format:
        checks.append(GateCheck(
            "crop", False,
            "wide-format source: crop derived from the dominant crop (not auto-approved)",
        ))
    elif crop.column:
        checks.append(GateCheck("crop", True, crop.column))
    else:
        checks.append(GateCheck(
            "crop", False, "no crop column (observations cannot be labeled by crop)",
        ))

    return checks


def gate_b(profile: SourceProfile) -> tuple[list[GateCheck], list[PlaceNameVerdict]]:
    """Name-safety checks: no unmapped or ambiguous place names.

    Returns ``(checks, suggested_aliases)``. New unambiguous fuzzy aliases are
    returned as suggestions; they never block auto-approval on their own.
    """
    suggested: list[PlaceNameVerdict] = []
    unmapped: list[str] = []
    ambiguous: list[str] = []
    for verdict in profile.place_verdicts:
        if verdict.status == "alias":
            suggested.append(verdict)
        elif verdict.status == "unmapped":
            unmapped.append(verdict.name)
        elif verdict.status == "ambiguous":
            ambiguous.append(verdict.name)

    checks: list[GateCheck] = []
    if unmapped:
        checks.append(GateCheck(
            "no-unmapped", False,
            f"{len(unmapped)} unmapped place name(s): "
            + ", ".join(sorted(unmapped)[:8])
            + (" ..." if len(unmapped) > 8 else ""),
        ))
    else:
        checks.append(GateCheck("no-unmapped", True, "all place names resolve"))

    if ambiguous:
        checks.append(GateCheck(
            "no-ambiguous", False,
            f"{len(ambiguous)} ambiguous place name(s): " + ", ".join(sorted(ambiguous)[:8]),
        ))
    else:
        checks.append(GateCheck("no-ambiguous", True, "no ambiguous place names"))

    return checks, suggested


def review(profile: SourceProfile) -> TableReview:
    """Run both gates over ``profile`` and collect warnings + alias suggestions."""
    checks_a = gate_a(profile)
    checks_b, suggested = gate_b(profile)
    return TableReview(
        profile=profile,
        gate_a=checks_a,
        gate_b=checks_b,
        warnings=list(profile.notes),
        suggested_aliases=suggested,
    )


def review_source(
    source: str | Path,
    *,
    vocab=None,
    **profiler_kwargs,
) -> TableReview:
    """Profile ``source`` and run both gates in one call."""
    return review(profile_tabular_source(source, vocab=vocab, **profiler_kwargs))


# --------------------------------------------------------------------------- #
# Wire-in helpers (stam.yaml tables list + name_aliases.py)
# --------------------------------------------------------------------------- #


_PLAIN_SAFE = re.compile(
    r"[^#&*!|>'\"%@`{}[\],:\s?-][^#&*!|>'\"%@`{}[\],:]*$"
)
_AMBIGUOUS_SCALARS = {
    "null", "~", "true", "false", "yes", "no", "on", "off",
    ".inf", "-.inf", ".nan", "",
}
_NUMBER_RE = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def _quote_str(value: str) -> str:
    """Serialize a string as a YAML plain/escaped scalar."""
    if value in _AMBIGUOUS_SCALARS or _NUMBER_RE.fullmatch(value):
        return json.dumps(value)
    if _PLAIN_SAFE.match(value):
        return value
    return json.dumps(value)


def _scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    if isinstance(value, str):
        return _quote_str(value)
    return str(value)


def _entry_yaml(entry: dict, indent: int = 4) -> str:
    """Render a tables entry as YAML matching the stam.yaml list style."""
    items = list(entry.items())
    lines = [" " * indent + "- " + items[0][0] + ": " + _scalar(items[0][1])]
    lines.extend(
        " " * (indent + 2) + key + ": " + _scalar(value)
        for key, value in items[1:]
    )
    return "\n".join(lines)


def _find_tables_block_end(lines: list[str], tables_line: int, indent: int) -> int | None:
    """First line after ``tables:`` where the next top-level block begins."""
    for index in range(tables_line + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if len(line) - len(line.lstrip()) < indent:
            return index
    return None


def append_table(stam_yaml: str | Path, entry: dict) -> str:
    """Append ``entry`` to ``tabular.tables`` in ``stam_yaml``.

    Returns ``"added"``, ``"duplicate"`` (same ``name`` already present) or
    ``"missing-tables"`` (no ``tabular.tables`` list to extend). Editing is a
    targeted text insertion that preserves comments and formatting.
    """
    path = Path(stam_yaml)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    tables_line = next(
        (i for i, line in enumerate(lines)
         if line.strip() == "tables:" and len(line) - len(line.lstrip()) == 2),
        None,
    )
    if tables_line is None:
        return "missing-tables"

    doc = yaml.safe_load(text)
    existing = (doc or {}).get("tabular", {}).get("tables") or []
    if any(str(table.get("name")) == str(entry.get("name")) for table in existing):
        return "duplicate"

    block = _entry_yaml(entry)
    block_end = _find_tables_block_end(lines, tables_line, indent=2)
    if block_end is None:
        new_text = text.rstrip("\n") + "\n\n" + block + "\n"
    else:
        lines.insert(block_end, block)
        new_text = "\n".join(lines) + "\n"
    path.write_text(new_text, encoding="utf-8")
    return "added"


def _existing_alias_keys(aliases_text: str) -> set[str]:
    match = re.search(r"ALIASES:\s*dict[^=]*=\s*\{", aliases_text)
    if match is None:
        return set()
    block = aliases_text[match.end():]
    block = block.split("}", 1)[0]
    keys = re.findall(r'^\s*["\']([^"\']+)["\']\s*:', block, re.MULTILINE)
    return {key_name(k) for k in keys}


def _strip_parenthetical(value: str) -> str:
    return value.split("(", 1)[0].strip() if "(" in value else value


def alias_suggestion(verdict: PlaceNameVerdict) -> tuple[str, str] | None:
    """``(source_name, canonical_boundary_name)`` for a suggested alias."""
    if not verdict.matches:
        return None
    canonical = _strip_parenthetical(clean_name(verdict.matches[0].name))
    if not canonical:
        return None
    return clean_name(verdict.name), canonical


def add_aliases(aliases_file: str | Path, additions: dict[str, str]) -> list[str]:
    """Insert missing ``(source -> canonical)`` pairs into the ALIASES dict.

    Returns the names actually added. Existing keys are left untouched.
    """
    path = Path(aliases_file)
    text = path.read_text(encoding="utf-8")
    existing = _existing_alias_keys(text)

    to_add: dict[str, str] = {}
    for name, value in additions.items():
        name, value = clean_name(name), clean_name(value)
        if not name or not value:
            continue
        if key_name(name) in existing:
            continue
        to_add[name] = value
    if not to_add:
        return []

    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ALIASES: dict"))
    close = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "}")
    inserted = [
        f"    {json.dumps(name)}: {json.dumps(value)}," for name, value in to_add.items()
    ]
    lines[close:close] = inserted
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list(to_add)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_review(review_: TableReview) -> None:
    print("Gate A (wiring sanity)")
    for check in review_.gate_a:
        print(f"  [{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    print("Gate B (name safety)")
    for check in review_.gate_b:
        print(f"  [{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    if review_.suggested_aliases:
        print("Suggested aliases (name -> canonical boundary name)")
        for verdict in review_.suggested_aliases:
            pair = alias_suggestion(verdict)
            if pair:
                print(f"  {pair[0]!r} -> {pair[1]!r}")
    if review_.warnings:
        print("Warnings")
        for note in review_.warnings:
            print(f"  - {note}")


def _cmd_list(args: argparse.Namespace) -> int:
    tabular_dir = Path(args.tabular_dir)
    if not tabular_dir.is_dir():
        print(f"tabular datasets dir not found: {tabular_dir}", file=sys.stderr)
        return 1
    sources = sorted(tabular_dir.glob("*.csv"))
    if not sources:
        print(f"no CSV files under {tabular_dir}")
        return 0
    for source in sources:
        try:
            review_ = review_source(source)
        except Exception as exc:  # keep listing the other files
            print(f"{source.name:<45} ERROR   {type(exc).__name__}: {exc}")
            continue
        verdict = "APPROVE" if review_.approved else "REVIEW"
        reasons = []
        for check in review_.gate_a + review_.gate_b:
            if not check.ok:
                reasons.append(f"{check.name}: {check.detail}")
        print(f"{source.name:<45} {verdict:<8} {' | '.join(reasons)}")
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    source = Path(args.path)
    if not source.is_file():
        print(f"source not found: {source}", file=sys.stderr)
        return 1

    print(f"Profiling {source}")
    review_ = review_source(source)
    _print_review(review_)

    if args.add_aliases and review_.suggested_aliases:
        additions = {
            name: canonical
            for verdict in review_.suggested_aliases
            for (name, canonical) in [alias_suggestion(verdict) or (None, None)]
            if name
        }
        added = add_aliases(args.aliases_file, additions)
        if added:
            print(f"added {len(added)} alias(es) to {args.aliases_file}")
        else:
            print("no new aliases to add")

    approved = review_.approved
    if not approved and not args.force:
        print("\nNOT auto-approved (see FAIL checks above).")
        print("Review the source, then either rerun with --force to wire it")
        print("anyway, or add the entry below to stam.yaml manually:")
        print()
        print(_entry_yaml(review_.entry))
        return 1

    if args.dry_run:
        print(f"\n[dry-run] would append {review_.profile.source.name} to {args.stam_yaml}")
        print(_entry_yaml(review_.entry))
        return 0

    result = append_table(args.stam_yaml, review_.entry)
    if result == "duplicate":
        print(f"{review_.profile.source.name} already present in {args.stam_yaml}")
        return 0
    if result == "missing-tables":
        print(f"{args.stam_yaml} has no tabular.tables list to extend", file=sys.stderr)
        return 1
    print(f"appended {review_.profile.source.name} to tabular.tables in {args.stam_yaml}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.stam.tabular_gates",
        description=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser(
        "list", help="profile every CSV under the tabular datasets dir and print verdicts"
    )
    list_parser.add_argument(
        "--tabular-dir", default=str(DEFAULT_TABULAR_DIR),
        help="directory scanned for CSVs (default: %(default)s)",
    )
    list_parser.set_defaults(handler=_cmd_list)

    add_parser = sub.add_parser(
        "add", help="profile one CSV and append it to stam.yaml when approved"
    )
    add_parser.add_argument("path", help="path to the CSV to wire in")
    add_parser.add_argument(
        "--stam-yaml", default=str(DEFAULT_STAM_YAML),
        help="stam.yaml to extend (default: %(default)s)",
    )
    add_parser.add_argument(
        "--aliases-file", default=str(DEFAULT_NAME_ALIASES),
        help="name_aliases.py to extend with --add-aliases (default: %(default)s)",
    )
    add_parser.add_argument(
        "--dry-run", action="store_true",
        help="print what would be written without writing anything",
    )
    add_parser.add_argument(
        "--force", action="store_true",
        help="append the table even when the review is not approved",
    )
    add_parser.add_argument(
        "--add-aliases", action="store_true",
        help="append suggested Gate-B aliases to name_aliases.py first",
    )
    add_parser.set_defaults(handler=_cmd_add)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
