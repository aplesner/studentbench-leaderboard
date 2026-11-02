"""Archive final analysis/validation source without SSOT, results or credentials."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def main(study=None):
    repository = Path(__file__).resolve().parents[2]
    output = Path(study) if study is not None else repository / "results/ranking-extension-20260911"
    output.mkdir(parents=True, exist_ok=True)
    files = [repository / name for name in ("README.md", "pyproject.toml", "uv.lock", ".gitignore")]
    for directory in ("src", "tests", "docs", "experiments"):
        files.extend(p for p in (repository / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".toml", ".txt", ".md", ".json"}
                     and "__pycache__" not in p.parts)
    inventory = {}
    archive = output / "source_snapshot.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as stream:
        for path in sorted(set(files)):
            if not path.is_file():
                continue
            relative = str(path.relative_to(repository))
            content = path.read_bytes()
            inventory[relative] = hashlib.sha256(content).hexdigest()
            info = ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            stream.writestr(info, content, compress_type=ZIP_DEFLATED, compresslevel=9)
    record = {"created_at": datetime.now(timezone.utc).isoformat(), "files": inventory,
              "n_files": len(inventory), "archive": archive.name,
              "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "scope": "Final code, experiment configurations, tests and documentation. Calls retain their individual runner hashes; transport/parser fixes occurred during execution, so this is the final reproducible validation/analysis source rather than every historical runner revision. Local SSOT, completed calls and the prepared human example are separate frozen artifacts."}
    (output / "source_snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"files": len(inventory), "archive_bytes": archive.stat().st_size,
                      "archive_sha256": record["archive_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, help="Artifact directory receiving the source archive")
    main(parser.parse_args().study)
