import pandas as pd
import fire
from typing import TypedDict, Set, List, Tuple
from pathlib import Path
import random

random.seed(42_2137)


class DatasetSplit(TypedDict):
    train: Set[str]
    test: Set[str]
    eval: Set[str]


DISEASES_TRAIN: Set[str] = set(["Diseases caused by viruses"])
DISEASES_TEST: Set[str] = set(
    ["Diseases caused by protozoa", "Diseases caused by parasites"]
)
DISEASES_EVAL: Set[str] = set(["Diseases caused by bacteria", "Fungal diseases"])
DISEASES_SPLIT: DatasetSplit = {
    "train": DISEASES_TRAIN,
    "test": DISEASES_TEST,
    "eval": DISEASES_EVAL,
}
assert all(
    not s1 & s2
    for i, s1 in enumerate(DISEASES_SPLIT.values())
    for s2 in list(DISEASES_SPLIT.values())[i + 1 :]
), "Diseases split has non-empty intersections - potential data-leakage."


def label_split(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Trying to split df with multiple categories into small subgroups each with same category"""
    grouped_df = df.groupby("category")
    categories = grouped_df.groups.keys()
    split = []
    for category in categories:
        category_df = df[df["category"] == category]
        max_idx = len(category_df) - 1
        beg_idx = 0
        end_idx = min(max_idx, beg_idx + random.randint(2, 3))
        while beg_idx < end_idx and end_idx <= max_idx:
            subset_df = category_df.iloc[beg_idx : end_idx + 1]
            split.append(subset_df)
            beg_idx = end_idx + 1
            end_idx = min(max_idx, beg_idx + random.randint(2, 3))
    return split


def split_diseases(dataset_dir, save_dir):
    assert (
        "diseases.json" in dataset_dir
    ), "This function can be only used to create splits for disesases dataset"
    dataset_dir = Path(dataset_dir)
    save_dir = Path(save_dir)
    df = pd.read_json(dataset_dir)
    split_dfs: List[Tuple[str, pd.DataFrame]] = [
        (
            save_dir / Path("diseases_" + split_name + ".json"),
            df[df["category"].isin(list(split_cols))],
        )
        for split_name, split_cols in DISEASES_SPLIT.items()
    ]
    for path, df in split_dfs:
        split_df = label_split(df)
        list_of_dicts_df = [df.to_dict(orient="records") for df in split_df]
        with open(path, "w") as json_file:
            json_file.write(
                pd.Series(list_of_dicts_df).to_json(orient="records", indent=2)
            )


def main(dataset_dir, save_dir):
    """If more datasets come, we need to choose them here"""
    split_diseases(dataset_dir, save_dir)


if __name__ == "__main__":
    fire.Fire(main)
