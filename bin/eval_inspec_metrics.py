from collections import defaultdict
import logging
import os
from typing import Any, Tuple, TypedDict

from dotenv import load_dotenv
import fire
import neptune
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.models.abstract_model import AbstractLabelModel
import tml_decoder.utils.common_utils as common_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


load_dotenv()

assert os.getenv("NEPTUNE_PROJECT") and os.getenv("NEPTUNE_API_TOKEN")

run = neptune.init_run(
    project=os.getenv("NEPTUNE_PROJECT"),
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
)


class ParsedDataset(TypedDict):
    train: Tuple[pd.DataFrame, pd.DataFrame]
    test: Tuple[pd.DataFrame, pd.DataFrame]
    eval: Tuple[pd.DataFrame, pd.DataFrame]


def eval_model(model: AbstractLabelModel, encoder: AbstractEncoder, parsed_dataset: ParsedDataset) -> Tuple[dict, dict]:
    result = {"train": {}, "test": {}, "eval": {}}
    predictions = {"train": {}, "test": {}, "eval": {}}

    for split_name, split in parsed_dataset.items():
        count_cos_sim_for_ground_truth = []
        count_cos_sim_for_avg_emb = []

        X = split[0]
        y = split[1]

        texts_for_summaries = defaultdict(list)

        # Iterate over the dataset and populate the defaultdict
        for text, summaries in zip(X, y):
            for summary in summaries:
                texts_for_summaries[summary].append(text)

        for summary, texts in tqdm(texts_for_summaries.items(), f"{split_name} split progress"):
            generated_label = model.get_label(texts)
            avg_embedding = encoder.average_embedding_for_texts(texts)

            true_label_embedding = encoder.encode(summary)
            generated_label_embedding = encoder.encode(generated_label)
            cos_sim_for_ground_truth = encoder.similarity(true_label_embedding, generated_label_embedding)
            cos_sim_for_avg_emb = encoder.similarity(avg_embedding, generated_label_embedding)
            count_cos_sim_for_ground_truth.append(cos_sim_for_ground_truth)
            count_cos_sim_for_avg_emb.append(cos_sim_for_avg_emb)

            logging.info(
                {
                    "category": summary,
                    "generated_label": generated_label,
                    "cos_sim_for_ground_truth": cos_sim_for_ground_truth,
                    "cos_sim_for_avg_emb": cos_sim_for_avg_emb,
                }
            )

            predictions[split_name][summary] = {
                "generated_label": generated_label,
                "cos_sim_for_ground_truth": cos_sim_for_ground_truth,
                "cos_sim_for_avg_emb": cos_sim_for_avg_emb,
            }

        assert len(count_cos_sim_for_ground_truth) > 0, f"Length of {split_name} is 0"
        average_cos_sim_for_gt = sum(count_cos_sim_for_ground_truth) / len(count_cos_sim_for_ground_truth)
        average_cos_sim_for_avg_emb = sum(count_cos_sim_for_avg_emb) / len(count_cos_sim_for_avg_emb)
        result[split_name]["avg_cos_sim_for_ground_truth"] = average_cos_sim_for_gt
        result[split_name]["avg_cos_sim_for_avg_emb"] = average_cos_sim_for_avg_emb

        if run is None:
            continue

        run[f"{split_name}/avg_cos_sim_for_ground_truth"] = average_cos_sim_for_gt
        run[f"{split_name}/avg_cos_sim_for_avg_emb"] = average_cos_sim_for_avg_emb

    return result, predictions


def read_dataset(path: str, random_state: int = 42) -> ParsedDataset:
    dataset = pd.read_json(path, lines=True)

    X = dataset["doc"]
    y = dataset["summary"]

    X_train, X_test_eval, y_train, y_test_eval = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_test, X_eval, y_test, y_eval = train_test_split(X_test_eval, y_test_eval, test_size=0.5, random_state=random_state)

    parsed_dataset: ParsedDataset = {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "eval": (X_eval, y_eval),
    }

    return parsed_dataset


def main(model_name: str, dataset_path: str, encoder_name: str, *args: Any, **kwargs: Any) -> None:
    run["dataset_path"] = dataset_path
    run["parameters"] = {
        "model_name": model_name,
        "encoder_name": encoder_name,
        "args": args,
        "kwargs": kwargs,
    }

    encoder = common_utils.get_encoder(encoder_name)
    model = common_utils.get_model(model_name, encoder, *args, **kwargs)
    dataset = read_dataset(dataset_path)

    results, predictions = eval_model(model, encoder, dataset)

    print(f"Metrics for {model.name}: ", results)

    run["results"] = results
    run["predictions"] = predictions
    run.stop()


if __name__ == "__main__":
    fire.Fire(main)
