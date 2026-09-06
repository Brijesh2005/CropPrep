"""Parse raw `kaggle kernels logs` text (v7) into the JSON-array log format and
extract the Phase 5/9 + Phase 3 + Phase 10 GPU diagnostic results.

Reads the text log captured from the CLI (one JSON payload per line, most lines
prefixed by a comma from the CLI's bracket-streaming), restores it into
``artifacts/r5_5_pull/logs_v7.json`` and prints the structured results.
"""

import json
import sys
from pathlib import Path

SRC = Path(r"C:\Users\brije\AppData\Local\Temp\opencode\log_v7.txt")
DST = Path(r"D:\CropPrep\artifacts\r5_5_pull\logs_v7.json")


def parse() -> list[dict]:
    raw = SRC.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    # entries are bracketed in a JSON array, each line a separate object
    cleaned = text.replace("\r", "").splitlines()
    payload = ""
    for ln in cleaned:
        payload += ln
    payload = payload.strip()
    if not payload.startswith("["):
        payload = "[" + payload
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # fallback: line-wise tolerate truncation
        arr = []
        for ln in cleaned:
            s = ln.strip().lstrip(",")
            if not s:
                continue
            try:
                arr.append(json.loads(s))
            except json.JSONDecodeError:
                continue
        return arr


def main() -> int:
    arr = parse()
    DST.write_text(json.dumps(arr), encoding="utf-8")
    print(f"parsed {len(arr)} log entries -> {DST}\n")

    lines = [(e.get("time", 0.0), e.get("stream_name", "?"), e.get("data", ""))
             for e in arr]

    def dump_span(start_marker: str, end_markers: tuple[str, ...],
                  label: str, max_len: int = 220000) -> None:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        capture = False
        buf: list[str] = []
        for _t, _s, d in lines:
            if start_marker in d:
                capture = True
                continue
            if capture and any(m in d for m in end_markers):
                break
            if capture:
                buf.append(d.strip())
        text = "\n".join(buf)
        i, j = text.find("{"), text.rfind("}")
        if 0 <= i < j:
            frag = text[i:j + 1]
            try:
                obj = json.loads(frag)
                s = json.dumps(obj, indent=2, default=str)
                print(s if len(s) <= max_len else s[:max_len] + "\n...[truncated]")
                return
            except Exception as ex:  # noqa: BLE001
                print("(json parse failed:", ex, ")")
        print(text[:max_len])

    dump_span("=== Phase 5/9", ("=== Phase 3",), "PHASE 5/9 - IMAGE SEPARABILITY + NORMALIZATION")
    dump_span("=== Phase 3", ("=== Phase 10",), "PHASE 3 - BINARY COCONUT-VS-PEPPER (image+tabular)")
    dump_span("=== Phase 10", ("=== Phase 11",), "PHASE 10 - FIRST-N-STEP DYNAMICS")

    print("\n=== FINAL SUMMARY ===")
    for _t, _s, d in lines[-120:]:
        if d.strip():
            print(d.rstrip())

    print("\n=== VARIANT KEY LINES (Phase 3) ===")
    for _t, _s, d in lines:
        if "variant:" in d or "prior_acc" in d or "acc=" in d and "macro_f1" in d and "beats" in d:
            print("  " + d.rstrip())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())