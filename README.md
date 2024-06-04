# TML-decoder

Team machine learning project for the University of Warsaw in collaboration with Huawei - Generative decoder for sentence embeddings.

## Setup

### Env

`.env` file contains secret keys, that shouldn't be shared. One should base on `.env.default` to create `.env` and populate it with the correct credentials.

```bash
cp .env.default .env
```

### Pre-commit

The project uses pre-commit hooks to ensure that code standards are maintained. The main configuration for pre-commit hooks is defined in `.pre-commit-config.yaml`. This file specifies the hooks for `ruff`, `black`, and `isort`. The configuration for these tools is located in the `pyproject.toml` file. Ensure you have pre-commit installed and set up by running:

```bash
pre-commit install
```
This will install the pre-commit hooks into your local repository, ensuring that they are run before each commit to maintain code quality and consistency.

### Neptune

https://app.neptune.ai/o/TML-Decoder/ is our project on Neptune. There one can find NEPTUNE_API_TOKEN for their account which should be put in the `.env` file.

### OpenAI

GPT-4 is used for generating summaries when creating a dataset. One needs to have an account on the OpenAI platform and generate an API key. The key should be put in the `.env` file.

## Working with the code

### Generating inspec dataset

First, you need to download inspec dump from link. For the purposes of our project we used the dump from 22nd March of 2024.

```bash
wget -P .dump https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/Inspec.zi
unzip -d .dump .dump/Inspec.zip
```

Then you can use `bin/create_inspec_dataset.py` script to generate dataset from the dump. The script will generate a dataset in the form of a JSON file with the following fields: `doc`, `keywords`, `summary`. The `keywords` column is a list of keywords that the article is described by. For the purpose of our project `keyword` is a category of the article, and each of them is described by corresponding `summary` item. Each `summary` is a summary done by GPT-4 of the sample of the articles that are described by the same `keyword`.

```bash
python bin/create_inspec_dataset.py --base_path='./.dump/Inspec' --sample_size=3 --seed=42
```

You can manipulate the `sample_size` parameter to control the number of samples that are used to generate the `summary` field, but please remember that many keywords have only a few samples, so setting `sample_size` to a high value may result in a lack of samples for some keywords.

### Evaluating models

```bash
python3 bin/eval_metrics.py --model_name='dumb' --dataset_name='diseases' --encoder_name='MiniLM'
```

To create your own model please inherit from abstract class in `abstract_model.py`.
To eval your own model / dataset / encoder please add it in `utils/common_utils.py`.
