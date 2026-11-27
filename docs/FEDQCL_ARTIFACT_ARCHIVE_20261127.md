# FedQCL artifact archive (2026-11-27)

This archive collects generated, non-model artifacts into the Kakarot checkout and publishes them to the `main` branch of `Anurag9000/FedQCL`.

## Included material

- `research_data/`: generated dataset shards and their metadata.
- `research_results/`: local experiment outputs, metrics, diagnostics, plots, and logs.
- `experiments/`: portable and resolved experiment ledgers, schedules, and manifests.
- `server_artifacts_sync/20261125/` and `server_artifacts_sync/20261127/`: collected server-side artifacts, organized by collection date and host.
- `campaign_archive/20261127/`: non-model campaign state, logs, receipts, and diagnostics.
- `campaign_operations.jsonl`: local campaign operation history.
- `github_artifacts/large_logs/`: lossless compressed replacements for non-model JSONL files larger than GitHub's regular 100 MiB file limit.
- `docs/FEDQCL_ARTIFACT_ARCHIVE_20261127.md` and `docs/FEDQCL_ARTIFACT_SHA256_20261127.jsonl`: archive notes and per-file integrity inventory.

The four remote hosts represented under the dated snapshot are Mtech, V100, L40 (`iiitd`), and H100 (`gpu2`). Files are retained in their source-relative directory structures.

## Exclusions

Model weights and checkpoints are excluded, including `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, and `.bin` files. Python bytecode, cache directories, virtual environments, temporary test outputs, and Git internals are also excluded. Credentials and private keys are not part of this archive.

The eight raw `communication_events.jsonl` files that exceed GitHub's regular per-file limit remain intact in the local artifact trees. Their `.gz` copies under `github_artifacts/large_logs/` are lossless substitutes for Git publication. The SHA-256 inventory records both the stored compressed file and its original source digest.

## Provenance and verification

The source checkout was at commit `856e82d24ee838485af4e2819113b30468d6ccb2` on `main`, matching `Anurag9000/FedQCL` at the start of publication. Remote snapshots were copied from the configured FedQCL checkouts using the authorized B550 route. File inventories were reconciled against each host. The large-log substitutions were gzip-tested and decompressed SHA-256 values were compared with the original files. The JSONL inventory records each included file's relative path, byte size, and SHA-256 digest; compressed substitutions additionally record their original path and original digest.

This archive is intended as research artifact storage. It does not claim that every recorded run is scientifically valid merely because its files are present; use each run's own status, provenance, and verification records when interpreting results.
