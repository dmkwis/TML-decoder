from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.models.abstract_model import AbstractLabelModel
import os
import neptune
import fire
from typing import Any, TypedDict
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import tml_decoder.utils.common_utils as common_utils

load_dotenv()

run = neptune.init_run(
    project=os.getenv("NEPTUNE_PROJECT"),
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
)


class ParsedDataset(TypedDict):
    train: list[pd.DataFrame]
    test: list[pd.DataFrame]
    split: list[pd.DataFrame]


def eval_model(
    model: AbstractLabelModel, encoder: AbstractEncoder, parsed_dataset: ParsedDataset
):
    result = {"train": {}, "test": {}, "eval": {}}
    for split_name, split in parsed_dataset.items():
        count_cos_sim_for_ground_truth = []
        count_cos_sim_for_avg_emb = []
        for subgroup in tqdm(split, f"{split_name} split progress"):
            true_label = subgroup["category"][0]
            texts = list(subgroup["title"])
            generated_label = model.get_label(texts)
            avg_embedding = encoder.average_embedding_for_texts(texts)

            true_label_embedding = encoder.encode(true_label)
            generated_label_embedding = encoder.encode(generated_label)
            cos_sim_for_ground_truth = encoder.similarity(
                true_label_embedding, generated_label_embedding
            )
            cos_sim_for_avg_emb = encoder.similarity(
                avg_embedding, generated_label_embedding
            )
            count_cos_sim_for_ground_truth.append(cos_sim_for_ground_truth)
            count_cos_sim_for_avg_emb.append(cos_sim_for_avg_emb)

            run[f"{split_name}/generated_label"].append(
                {
                    "category": true_label,
                    "generated_label": generated_label,
                    "cos_sim_for_ground_truth": cos_sim_for_ground_truth,
                    "cos_sim_for_avg_emb": cos_sim_for_avg_emb,
                }
            )

        assert len(count_cos_sim_for_ground_truth) > 0, f"Length of {split_name} is 0"
        average_cos_sim_for_gt = sum(count_cos_sim_for_ground_truth) / len(
            count_cos_sim_for_ground_truth
        )
        average_cos_sim_for_avg_emb = sum(count_cos_sim_for_avg_emb) / len(
            count_cos_sim_for_avg_emb
        )
        result[split_name]["avg_cos_sim_for_ground_truth"] = average_cos_sim_for_gt
        result[split_name]["avg_cos_sim_for_avg_emb"] = average_cos_sim_for_avg_emb
        run[f"{split_name}/avg_cos_sim_for_ground_truth"] = average_cos_sim_for_gt
        run[f"{split_name}/avg_cos_sim_for_avg_emb"] = average_cos_sim_for_avg_emb

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


def main(
    model_name: str, dataset_name: str, encoder_name: str, *args: Any, **kwargs: Any
) -> None:
    run["dataset_name"] = dataset_name
    run["parameters"] = {
        "model_name": model_name,
        "encoder_name": encoder_name,
        "args": args,
        "kwargs": kwargs,
    }  # TODO There should be parameters of the model like hyperparameters

    encoder = common_utils.get_encoder(encoder_name)
    model = common_utils.get_model(model_name, encoder, *args, **kwargs)
    dataset = read_dataset(dataset_name)

    results = eval_model(model, encoder, dataset)

    print(f"metrics for {model.name}: ", results)

    run["results"] = results
    run.stop()


if __name__ == "__main__":
    fire.Fire(main)
