from collections import defaultdict
import logging
import os
from typing import Any, List, Optional, Tuple, TypedDict

import neptune
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.generators.abstract_generator import AbstractGenerator
from tml_decoder.metrics import Metrics
from tml_decoder.models.abstract_model import AbstractLabelModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ParsedDataset(TypedDict):
    train: Tuple[pd.DataFrame, pd.DataFrame]
    test: Tuple[pd.DataFrame, pd.DataFrame]
    eval: Tuple[pd.DataFrame, pd.DataFrame]


def initialize_neptune_run():
    """Initialize Neptune run with environment variables."""
    return neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )


def eval_model(
    model: AbstractLabelModel,
    encoder: AbstractEncoder,
    generator: AbstractGenerator,
    parsed_dataset: ParsedDataset,
    run: Any,
    metrics_to_skip: Optional[List[str]] = None,
    batch_size: int = 16,
) -> Tuple[dict, dict]:
    result = {"train": {}, "test": {}, "eval": {}}
    predictions = {"train": {}, "test": {}, "eval": {}}

    for split_name, split in parsed_dataset.items():
        X = split[0]
        y = split[1]

        true_labels = []
        generated_labels = []
        texts = []
        reference_texts = []
        generated_texts = []
        reference_summaries = []
        generated_summaries = []

        texts_for_summaries = defaultdict(list)

        for text, summaries in zip(X, y):
            for summary in summaries:
                texts_for_summaries[summary].append(text)

        for summary, texts_group in tqdm(texts_for_summaries.items(), f"{split_name} split progress"):
            generated_label = model.get_label(texts_group)

            true_labels.append(summary)
            generated_labels.append(generated_label)
            texts.append(texts_group)
            reference_texts.append(" ".join(texts_group))
            generated_texts.append(generated_label)
            reference_summaries.append(summary)
            generated_summaries.append(generated_label)

            logging.info(
                {
                    "category": summary,
                    "generated_label": generated_label,
                }
            )

            predictions[split_name][summary] = {
                "generated_label": generated_label,
                "texts": texts_group,
                "category": summary,
            }

        metrics = Metrics(encoder, generator, batch_size=batch_size, metrics_to_skip=metrics_to_skip)
        metrics_result = metrics.calculate_metrics(true_labels, generated_labels, texts, reference_texts, generated_texts, reference_summaries, generated_summaries)
        result[split_name] = metrics_result

        if run is None:
            continue

        for metric_name, metric_value in metrics_result.items():
            run[f"{split_name}/{metric_name}"] = metric_value

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
