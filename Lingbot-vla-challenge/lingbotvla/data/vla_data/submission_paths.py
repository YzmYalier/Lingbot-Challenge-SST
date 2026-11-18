"""Keep raw-teacher metadata relocatable when running a submission recipe."""
import os
from pathlib import Path


def raw_source_path(source, dataset_root):
    if not source:
        raise ValueError(f"No raw teacher source in {dataset_root}/meta/sam_info.json")
    if os.environ.get("SUBMISSION_ROOT"):
        from submit_paths import base_path, path, resolve
        if source.startswith("submission://"):
            name, _, suffix = source[len("submission://"):].partition("/")
            result = (path(name) / suffix).resolve()
        else:
            result = resolve(source)
        # process_data.sh records an absolute source at generation time. When
        # the entire bundle moves, rebind only this configured SAM/raw pair.
        if not result.is_relative_to(base_path()):
            if Path(dataset_root).resolve() == path("sam_output") and result.name == path("raw").name:
                result = path("raw")
            else:
                raise ValueError(f"Raw teacher source is outside the submission: {result}")
        return result
    source = Path(source).expanduser()
    return source if source.is_absolute() else Path(dataset_root) / source
