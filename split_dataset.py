from pathlib import Path
import numpy as np
import shutil
from tqdm import tqdm


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
tolerance = 1e-9  # to deal with floating point "imprecision"
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1) < tolerance,\
    f"Train, validation, and test fractions must add up to 1."


SRC_IMAGE_DIR = Path(r"data/raw/images/Images")
SPLIT_DIR = Path(r"data/split")
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"
TRAIN_DIR.mkdir()
VAL_DIR.mkdir()
TEST_DIR.mkdir()


for class_dir in tqdm(
    SRC_IMAGE_DIR.glob(r"*"),
    "Classes progress",
    len(list(SRC_IMAGE_DIR.glob(r"*")))
):

    # find out the number of images in a specific breed-directory
    images_list = np.array(list(class_dir.glob(r"*.jpg")))
    num_samples = len(images_list)

    # construct train/val/test mask
    # initally fill with "train"
    mask = np.repeat("train", num_samples)
    # mark leftmost samples as "validation"
    mask[:int(num_samples*VAL_FRAC)] = "val"
    # mark rightmost samples as "test"
    mask[-int(num_samples*TEST_FRAC):] = "test"
    # shuffle in-place
    np.random.shuffle(mask)

    # copy images to the corresponding split's subfolder
    for split_dir, tag in zip([TRAIN_DIR, VAL_DIR, TEST_DIR],
                           ["train", "val", "test"]):
        split_class_dir = split_dir / class_dir.name
        if not split_class_dir.exists():
            split_class_dir.mkdir()
        for image in images_list[mask==tag]:
            shutil.copy(image, split_class_dir / image.name)
