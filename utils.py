import shutil
from pathlib import Path
import json


def save_code(
        src_dir, dst_dir,
        patterns=[
            "*.py",
            "*.ipynb",
            "params.json"
        ],
        iterative=False
):
    # TODO: implement iterative (tho don't really need in my case)
    
    for pattern in patterns:
        for path in src_dir.glob(pattern):
            # shutil.copy2(str(src_dir / path), str(dst_dir / path.name))
            shutil.copy2(src_dir / path, dst_dir / path.name)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()  # Ensure immediate flushing of text

    def flush(self):
        for file in self.files:
            file.flush()


def make_unique_dir_name(dir):
    """Renames the run name till it's unique. Returns original if it is
    initially unique
    """
    orig_run_name = dir.name
    copy_idx = 0
    while dir.exists():
        dir = dir.with_name(orig_run_name + f"_copy{copy_idx}")
        copy_idx += 1
    
    return dir


def load_json(path):
    path = Path(path)
    with path.open("r") as f:
        result = json.load(f)
    
    return result