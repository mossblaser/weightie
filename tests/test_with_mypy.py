import subprocess

from pathlib import Path

import weightie


def test_mypy():
    subprocess.run(
        [
            "mypy",
            str(Path(weightie.__file__).parent),
            str(Path(__file__).parent),
        ],
        check=True,
    )
