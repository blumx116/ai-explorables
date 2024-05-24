import json
import os
import sys
from os import statvfs_result
from os.path import isdir
from pathlib import Path
from typing import Optional

from safetensors import safe_open


def safetensors_to_json_obj(safetensors) -> dict:
    return {key: safetensors.get_tensor(key).tolist() for key in safetensors.keys()}


def convert_file(input_path: Path, output_path: Optional[Path] = None) -> None:
    if not input_path.suffix == "safetensors":
        input_path = input_path / "model.safetensors"
    if output_path is None:
        output_path = Path("model.json")
    elif isdir(output_path):
        output_path = output_path / "model.json"

    safetensors = safe_open(input_path, framework="torch")

    jsontensors: dict = safetensors_to_json_obj(safetensors)
    with open(output_path, "w") as f:
        json.dump(jsontensors, f)


if __name__ == "__main__":
    output_path: Optional[Path] = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    convert_file(Path(sys.argv[1]), output_path)
