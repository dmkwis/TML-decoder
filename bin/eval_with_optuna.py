import logging
import os
from typing import Any, List

from dotenv import load_dotenv
import fire
import optuna

from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.eval_functions import eval_model, initialize_neptune_run, read_dataset
from tml_decoder.generators.abstract_generator import AbstractGenerator
import tml_decoder.utils.common_utils as common_utils
from tml_decoder.models.mcts_model import MCTSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

assert os.getenv("NEPTUNE_PROJECT") and os.getenv("NEPTUNE_API_TOKEN")

initial_prompts = ["This cluster describe", "These documents are about"]

def objective(trial: optuna.Trial, dataset_path: str, model_name: str, encoder: AbstractEncoder, generator: AbstractGenerator, study_name: str) -> float:
    run = initialize_neptune_run()
    run["parameters/trial_number"] = trial.number
    run["parameters/study_name"] = study_name

    # Example hyperparameter space
    exploration_weight = trial.suggest_float("exploration_weight", 0.5, 2.0)
    perplexity_weight = trial.suggest_float("perplexity_weight", 1e-5, 1e-3)
    initial_prompt = trial.suggest_categorical("initial_prompt", initial_prompts)

    model = MCTSModel(
        encoder=encoder,
        generator=generator,
        guide=common_utils.get_guide("random", encoder),
        iter_num=10,
        max_len=100,
        min_result_len=40,
        initial_prompt=initial_prompt,
        exploration_weight=exploration_weight,
        perplexity_weight=perplexity_weight,
    )

    dataset = read_dataset(dataset_path)

    results, predictions = eval_model(model, encoder, generator, dataset, run)

    objective_value = results["eval"]["cosine_similarity"]["cos_sim_for_avg_emb"]

    run["results"] = results
    run["predictions"] = predictions
    run["objective_value"] = objective_value

    run.stop()

    return objective_value


def tune_hyperparameters(
    study_name: str, model_name: str, dataset_path: str, encoder_name: str, direction: str = "minimize", n_trials: int = 100, *args: Any, **kwargs: Any
) -> None:

    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction=direction, load_if_exists=True)
    
    encoder = common_utils.get_encoder(encoder_name)
    generator = common_utils.get_generator("gpt2")

    study.optimize(lambda trial: objective(trial, dataset_path, model_name, encoder, generator, study_name), n_trials=n_trials)

    logging.info("Best hyperparameters: ", study.best_params)
    logging.info("Best objective value: ", study.best_value)


if __name__ == "__main__":
    fire.Fire(tune_hyperparameters)