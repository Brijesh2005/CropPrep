"""Patch training/dataset_manager/providers/kaggle_image.py so the imagery
provider reads the attached Kaggle dataset **in place** from
/kaggle/input/... instead of downloading or copying the ~137GB dataset into
the repo's working directory.

Run this from the repository root (the folder that contains `training/`):

    python patch_kaggle_image_mount.py

Or point it at a specific repo:

    python patch_kaggle_image_mount.py --repo-root /path/to/CropPrep

It is idempotent — running it twice is safe; it detects whether the patch is
already applied and skips it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET_REL_PATH = "training/dataset_manager/providers/kaggle_image.py"

OLD_IMPORT_BLOCK = '''from __future__ import annotations

from datetime import datetime'''

NEW_IMPORT_BLOCK = '''from __future__ import annotations

import os
from datetime import datetime'''

OLD_METHOD = '''        if self._mounted_root is None:
            slug = self.handle.split("/")[-1]
            candidate = Path("/kaggle/input") / slug
            self._mounted_root = candidate if candidate.is_dir() else None
        return self._mounted_root'''

NEW_METHOD = '''        if self._mounted_root is None:
            owner, _, slug = self.handle.partition("/")
            candidates: list[Path] = []
            # Explicit override always wins (e.g. a non-standard mount path).
            env_root = os.environ.get("CROPFUSION_KAGGLE_IMAGE_ROOT")
            if env_root:
                candidates.append(Path(env_root))
            candidates += [
                # Classic "Add Data" mount: /kaggle/input/<dataset-slug>
                Path("/kaggle/input") / slug,
                # kagglehub-style mount seen on some Kaggle runtimes:
                # /kaggle/input/datasets/<owner>/<dataset-slug>
                Path("/kaggle/input/datasets") / owner / slug,
                Path("/kaggle/input/datasets") / self.handle,
            ]
            self._mounted_root = next(
                (c for c in candidates if c.is_dir()), None
            )
        return self._mounted_root'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the CropPrep repository root (default: current directory)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    target = repo_root / TARGET_REL_PATH

    if not target.is_file():
        print(f"ERROR: could not find {target}", file=sys.stderr)
        print(
            "Pass --repo-root pointing at the CropPrep checkout "
            "(the folder containing 'training/').",
            file=sys.stderr,
        )
        return 1

    text = target.read_text(encoding="utf-8")
    original_text = text
    changed = False

    # Already patched?
    if "CROPFUSION_KAGGLE_IMAGE_ROOT" in text:
        print(f"[skip] {target} already patched.")
        return 0

    if OLD_IMPORT_BLOCK in text:
        text = text.replace(OLD_IMPORT_BLOCK, NEW_IMPORT_BLOCK, 1)
        changed = True
    elif "import os" not in text.splitlines()[0:10]:
        # Fallback: insert "import os" right after the __future__ import line
        # if the exact block above doesn't match (e.g. file already edited).
        lines = text.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.strip() == "from __future__ import annotations":
                if "import os\n" not in lines[i + 1 : i + 3]:
                    lines.insert(i + 1, "\n")
                    lines.insert(i + 2, "import os\n")
                    text = "".join(lines)
                    changed = True
                break

    if OLD_METHOD in text:
        text = text.replace(OLD_METHOD, NEW_METHOD, 1)
        changed = True
    else:
        print(
            "WARNING: could not find the exact _kaggle_input_root() method "
            "body to replace (file may already differ from the expected "
            "version). No method-level change applied.",
            file=sys.stderr,
        )

    if not changed or text == original_text:
        print(
            "ERROR: no changes were applied — the file contents did not "
            "match what this script expects. Please check the file "
            "manually.",
            file=sys.stderr,
        )
        return 1

    backup = target.with_suffix(target.suffix + ".bak")
    backup.write_text(original_text, encoding="utf-8")
    target.write_text(text, encoding="utf-8")

    print(f"[ok] patched {target}")
    print(f"[ok] backup saved to {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
