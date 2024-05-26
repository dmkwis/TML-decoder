import logging
from typing import Any, List, Optional

from dotenv import load_dotenv
import fire

from tml_decoder.eval_functions import eval_model, initialize_neptune_run, read_dataset
import tml_decoder.utils.common_utils as common_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

run = initialize_neptune_run()


def main(model_name: str, dataset_path: str, encoder_name: str, batch_size: int, metrics_to_skip: Optional[List[str]] = None, *args: Any, **kwargs: Any) -> None:
    run["dataset_path"] = dataset_path
    run["parameters"] = {
        "model_name": model_name,
        "encoder_name": encoder_name,
        "args": args,
        "kwargs": kwargs,
    }

    encoder = common_utils.get_encoder(encoder_name)
    generator = common_utils.get_generator("gpt2")
    model = common_utils.get_model(model_name, encoder, *args, **kwargs)
    dataset = read_dataset(dataset_path)

    results, predictions = eval_model(model, encoder, generator, dataset, run, metrics_to_skip, batch_size)

    print(f"Metrics for {model.name}: ", results)

    run["results"] = results
    run["predictions"] = predictions
    run.stop()


if __name__ == "__main__":
    fire.Fire(main)
