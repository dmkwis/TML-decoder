from typing import Dict, List, Tuple
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.models.abstract_model import AbstractLabelModel
import os
import neptune
import fire
from typing import Any, TypedDict
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

import tml_decoder.utils.common_utils as common_utils
from tml_decoder.metrics.rouge_n import evaluate_rouge_n
from tml_decoder.metrics.perplexity_evaluator import evaluate_perplexity
from tml_decoder.metrics.cosine_similarity import evaluate_cosine_similarity

load_dotenv()

run = neptune.init_run(
    project=os.getenv("NEPTUNE_PROJECT"),
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
)


class ParsedDataset(TypedDict):
    train: list[pd.DataFrame]
    test: list[pd.DataFrame]
    split: list[pd.DataFrame]

def evaluate_labels_and_embeddings(model: AbstractLabelModel, encoder: AbstractEncoder, split: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[List[str]]]:
    true_labels = []
    generated_labels = []
    all_texts = []
    for subgroup in split:
        true_labels.append(subgroup["category"][0])
        texts = list(subgroup["title"])
        all_texts.append(texts)
        generated_label = model.get_label(texts)
        generated_labels.append(generated_label)
    return true_labels, generated_labels, all_texts

def compile_results(split_name: str, cos_sim_results: Dict[str, List[float]], rouge_scores: Dict[str, float], perplexities: Tuple[List[float], List[float]]) -> Dict[str, Any]:
    result = {}
    result[split_name] = {
        "avg_cos_sim_for_ground_truth": np.mean(cos_sim_results["cos_sim_for_ground_truth"]),
        "avg_cos_sim_for_avg_emb": np.mean(cos_sim_results["cos_sim_for_avg_emb"]),
        "rouge_scores": rouge_scores,
        "avg_perplexity_reference": np.mean(perplexities[0]),
        "avg_perplexity_generated": np.mean(perplexities[1]),
    }
    return result

def eval_model(model: AbstractLabelModel, encoder: AbstractEncoder, parsed_dataset: Dict[str, Any], model_name: str = 'gpt2'):
    result = {}
    for split_name, split in parsed_dataset.items():
        true_labels, generated_labels, all_texts = evaluate_labels_and_embeddings(model, encoder, split)
        
        cos_sim_results = evaluate_cosine_similarity(encoder, true_labels, generated_labels, all_texts)
        rouge_scores = evaluate_rouge_n(true_labels, generated_labels)
        perplexities = evaluate_perplexity(true_labels, generated_labels, model_name=model_name)

        result.update(compile_results(split_name, cos_sim_results, rouge_scores, perplexities))
    
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
