# FedQCL artifact archive (2026-11-27)

This archive collects non-dataset, non-model experiment artifacts into the Kakarot checkout and publishes them to the `main` branch of `Anurag9000/FedQCL`.

## Included material

- `research_results/`: experiment outputs, metrics, diagnostics, plots, and logs.
- `experiments/`: portable and resolved experiment ledgers, schedules, and manifests.
- `server_artifacts_sync/20261125/` and `server_artifacts_sync/20261127/`: collected server-side artifacts, organized by collection date and host.
- `campaign_archive/20261127/`: non-model, non-dataset campaign state, logs, receipts, and diagnostics.
- `campaign_operations.jsonl`: campaign operation history.
- `github_artifacts/large_logs/`: lossless compressed replacements for JSONL files larger than GitHub's regular 100 MiB file limit.
- `docs/FEDQCL_ARTIFACT_ARCHIVE_20261127.md` and `docs/FEDQCL_ARTIFACT_SHA256_20261127.jsonl`: archive notes and per-file integrity inventory.

The dated server snapshot represents Mtech, V100, L40 (`iiitd`), and H100 (`gpu2`). Files retain their source-relative directory structures.

## Exclusions

Dataset files and model weights/checkpoints are excluded, including `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, and `.bin` files. Python bytecode, cache directories, virtual environments, temporary test outputs, credentials, private keys, and Git internals are also excluded. Existing local dataset directories are left on the Kakarot machine but are not part of the current published tree.

The eight raw `communication_events.jsonl` files exceeding GitHub's regular per-file limit remain intact in local result trees. Their `.gz` copies under `github_artifacts/large_logs/` are lossless publication substitutes. The SHA-256 inventory records the original path and digest alongside the stored compressed path, size, and digest.

## Provenance and verification

The source checkout was at commit `856e82d24ee838485af4e2819113b30468d6ccb2` on `main`, matching `Anurag9000/FedQCL` at the start of publication. Remote result, campaign, and experiment trees were copied through the authorized B550 route and inventoried against their source hosts. The large-log substitutions were gzip-tested and their decompressed SHA-256 values were compared with the original files. The JSONL inventory records each included file's relative path, byte size, and SHA-256 digest, plus original-to-compressed mappings for the eight substitutions.

After the request was clarified to exclude datasets, dataset files were removed from the current `main` tree in a follow-up commit; they remain in earlier Git history because the branch was not rewritten. The partial additional source-dataset transfer was deleted locally before publication.

Presence in this archive does not by itself establish that a run is scientifically valid. Use each run's own status and provenance when interpreting results.
