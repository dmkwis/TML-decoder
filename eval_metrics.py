from abstract_encoder import AbstractEncoder
from abstract_model import AbstractLabelModel
import os
import neptune
import fire
from typing import TypedDict
import pandas as pd
from dotenv import load_dotenv

import common_utils
from abstract_model import AbstractLabelModel

load_dotenv()


class ParsedDataset(TypedDict):
    train: list[pd.DataFrame]
    test: list[pd.DataFrame]
    split: list[pd.DataFrame]


def eval_model(
    model: AbstractLabelModel, encoder: AbstractEncoder, parsed_dataset: ParsedDataset
):
    result = {"train": {}, "test": {}, "eval": {}}
    for split_name, split in parsed_dataset.items():
        count_cos_sim = []
        for subgroup in split:
            true_label = subgroup["category"][0]
            texts = list(subgroup["title"])
            generated_label = model.get_label(texts)
            true_label_embedding = encoder.encode(true_label)
            generated_label_embedding = encoder.encode(generated_label)
            cos_sim = encoder.similarity(
                true_label_embedding, generated_label_embedding
            )
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
            raise FileNotFoundError(f"File {path} not found")
        reconstructed_dataframes = [pd.DataFrame(data) for data in dataframes_as_dicts]
        parsed_dataset[split_name] = reconstructed_dataframes
    return parsed_dataset


def main(model_name: str, dataset_name: str, encoder_name: str) -> None:
    run = neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )
    model = None
    dataset = None
    encoder = common_utils.get_encoder(encoder_name)
    model = common_utils.get_model(model_name, encoder=encoder)
    dataset = read_dataset(dataset_name)
    results = eval_model(model, encoder, dataset)
    print(f"metrics for {model.name}: ", results)

    run["dataset_name"] = dataset_name
    run["parameters"] = {"model_name": model.name} # TODO There should be parameters of the model like hyperparameters
    run["eval"] = results
    run.stop()


if __name__ == "__main__":
    fire.Fire(main)
