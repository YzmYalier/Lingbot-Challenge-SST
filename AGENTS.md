# Submission maintenance rules

- Limit changes to this submission directory. Treat the original workspace as read-only.
- Resolve project data, models, interpreters, outputs and caches through the root `base.json` and `submit_paths.py`. Do not add machine-specific absolute paths to source code or launch scripts.
- A script may use `__file__` to locate `base.json` or import the resolver; its project paths must then come from that configuration.
- Keep generated logs, temporary files and caches inside this directory. Do not create hard links or external symlinks back into the original workspace.
- Preserve the 2500 episode IDs, all numerical state/action data, and the order of the 100 seen instructions per episode. Never publish a subset as either requested full dataset.
- The maintained inference client is `inference/`; `RoboTwin/script/deploy/sam_mask/` only exposes compatibility imports.
- Do not launch full GPU data regeneration as a routine code test. Use the included logic tests and a separate CPU smoke output; retain explicit validation limitations in the README/report.
- Vendor source examples outside the documented data/mask entry points remain upstream references. When bringing one into an executable submission workflow, migrate its local paths to `base.json` first.
