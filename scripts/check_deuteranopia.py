"""Write a PNG as a deuteranope sees it: `python scripts/check_deuteranopia.py in.png out.png`."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from hiring_bias_xai import plotting


def main(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        seen = plotting.simulate_deuteranopia(np.array(image.convert("RGBA")))
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(seen).save(target)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
