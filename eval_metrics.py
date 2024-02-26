from abstract_model import AbstractLabelModel
from dumb_model import DumbModel
import fire
from typing import TypedDict
import pandas as pd
import common_utils


class ParsedDataset(TypedDict):
    train: list[pd.DataFrame]
    test: list[pd.DataFrame]
    split: list[pd.DataFrame]


def eval_model(model: AbstractLabelModel, parsed_dataset: ParsedDataset):
    result = {"train": {}, "test": {}, "eval": {}}
    for split_name, split in parsed_dataset.items():
        count_cos_sim = []
        for subgroup in split:
            true_label = subgroup["category"][0]
            texts = list(subgroup["title"])
            generated_label = model.get_label(texts)
            cos_sim = common_utils.embedder.encode(
                true_label
            ) @ common_utils.embedder.encode(generated_label)
            count_cos_sim.append(cos_sim)
        assert len(count_cos_sim) > 0, f"Length of {split_name} is 0"
        average_cos_sim = sum(count_cos_sim) / len(count_cos_sim)
        result[split_name]["avg_cos_sim"] = average_cos_sim
    return result


def read_dataset(dataset_name: str) -> ParsedDataset:
    paths = {
        "train": f"dataset_split/{dataset_name}_train.json",
        "test": f"dataset_split/{dataset_name}_test.json",
        "eval": f"dataset_split/{dataset_name}_eval.json",
    }
    parsed_dataset = {}
    for split_name, path in paths.items():
        try:
            with open(path, "r") as json_file:
                dataframes_as_dicts = pd.read_json(json_file, typ="series")
        except IOError:
            return None
        reconstructed_dataframes = [pd.DataFrame(data) for data in dataframes_as_dicts]
        parsed_dataset[split_name] = reconstructed_dataframes
    return parsed_dataset


def main(model_name: str, dataset_name: str):
    model = None
    dataset = None
    if model_name == "dumb":
        model = DumbModel()
    assert model is not None, f"Can't find model with name {model_name}"
    dataset = read_dataset(dataset_name)
    assert dataset is not None, f"Can't find dataset with name {dataset_name}"
    results = eval_model(model, dataset)
    print(f"metrics for {model.name}: ", results)


if __name__ == "__main__":
    fire.Fire(main)
