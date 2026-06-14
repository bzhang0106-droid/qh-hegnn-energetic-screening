# Code package audit (2026-06-09)

Scope: NPJ_submission/code was audited against the local historical code-release snapshot and remote-script snapshot. The thread index did not expose an older readable work thread, so the local historical release artifacts were used as the authoritative reconstruction source.

## Source snapshots used

- Manuscript/06_code_release: main historical code release, including workflow source, final-freeze automation, SLURM templates, and manuscript/QTAIM generation scripts.
- codex_remote_scripts: remote-side scripts and SLURM jobs used for final npj/QTAIM validation runs.
- Existing NPJ_submission/code: current package scripts were treated as newer/manual-edited when their hashes differed and were not overwritten.

## Copy policy

- Added missing source/script/readme files only: .py, .R, .sh, .slurm, .ps1, .cmd, .md, .txt.
- Did not copy binary artifacts, model files, figures, PDFs, Word files, caches, logs, or result folders.
- Preserved current package edits by skipping existing files when target content differed.
- Removed generated __pycache__ directories from the submission code folder.

## Audit result

- Missing files copied into the code package: 127
- Existing files identical to historical source and left unchanged: 2
- Existing files kept because the current package version differed: 1
- Generated cache directories removed: 1

Detailed per-file records:

- CODE_PACKAGE_MANIFEST_20260609.csv: final file manifest with SHA256 hashes.
- CODE_PACKAGE_ACTION_LOG_20260609.csv: copy/skip/removal audit log.
