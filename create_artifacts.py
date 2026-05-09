from __future__ import annotations

import shutil
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_BUNDLE_DIR = WORKSPACE_ROOT / "Procesare Dataset" / "deployment" / "current" / "bundle"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "model_artifacts"
REQUIRED_FILES = (
    "model.pt",
    "scaler.pkl",
    "feature_columns.json",
    "bank_mapping_rules.yaml",
    "thresholds.json",
    "model_metadata.json",
)


def _ensure_source_bundle() -> None:
    if not CANONICAL_BUNDLE_DIR.is_dir():
        raise FileNotFoundError(
            "Missing canonical inference bundle: "
            f"{CANONICAL_BUNDLE_DIR}"
        )

    missing = [name for name in REQUIRED_FILES if not (CANONICAL_BUNDLE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Canonical inference bundle is incomplete. Missing files: "
            f"{missing}"
        )


def _sync_bundle(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_names = set(REQUIRED_FILES)

    for existing in dst_dir.iterdir():
        if existing.is_file() and existing.name not in src_names:
            existing.unlink()

    for name in REQUIRED_FILES:
        shutil.copy2(src_dir / name, dst_dir / name)


def main() -> None:
    _ensure_source_bundle()
    _sync_bundle(CANONICAL_BUNDLE_DIR, ARTIFACTS_DIR)
    print(f"Mirrored canonical bundle from {CANONICAL_BUNDLE_DIR} to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
