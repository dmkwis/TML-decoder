import logging
import os
from typing import Any

from dotenv import load_dotenv
import fire
import optuna

from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.eval_functions import eval_model, initialize_neptune_run, read_dataset
from tml_decoder.generators.abstract_generator import AbstractGenerator
import tml_decoder.utils.common_utils as common_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

assert os.getenv("NEPTUNE_PROJECT") and os.getenv("NEPTUNE_API_TOKEN")


def objective(trial: optuna.Trial, dataset_path: str, model_name: str, encoder: AbstractEncoder, generator: AbstractGenerator, study_name: str) -> float:
    run = initialize_neptune_run()
    run["parameters/trial_number"] = trial.number
    run["parameters/study_name"] = study_name

    # Example hyperparameter space
    param1 = trial.suggest_float("param1", 0.0, 1.0)
    param2 = trial.suggest_int("param2", 1, 100)

    model = common_utils.get_model(model_name, encoder, param1=param1, param2=param2)
    dataset = read_dataset(dataset_path)

    results, predictions = eval_model(model, encoder, generator, dataset, run)

    # Example: using average cosine similarity on test split as the objective value
    objective_value = results["val"]["cosine_similarity"]["avg_cos_sim_for_ground_truth"]

    run["results"] = results
    run["predictions"] = predictions
    run["objective_value"] = objective_value

    run.stop()

    return objective_value


def tune_hyperparameters(
    study_name: str, model_name: str, dataset_path: str, encoder_name: str, direction: str = "minimize", n_trials: int = 100, *args: Any, **kwargs: Any
) -> None:
    study = optuna.create_study(direction=direction)

    encoder = common_utils.get_encoder(encoder_name)
    generator = common_utils.get_generator("gpt2")

    study.optimize(lambda trial: objective(trial, dataset_path, model_name, encoder, generator, study_name), n_trials=n_trials)

    logging.info("Best hyperparameters: ", study.best_params)
    logging.info("Best objective value: ", study.best_value)


if __name__ == "__main__":
    fire.Fire(tune_hyperparameters)
