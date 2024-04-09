# TML-decoder

Team machine learning project for the University of Warsaw in collaboration with Huawei - Generative decoder for sentence embeddings.

## Setup

### Env

`.env` file contains secret keys, that shouldn't be shared. One should base on `.env.default` to create `.env` and populate it with the correct credentials.

```bash
cp .env.default .env
```

### Neptune

https://app.neptune.ai/o/TML-Decoder/ is our project on Neptune. There one can find NEPTUNE_API_TOKEN for their account.

## Working with the code

### Generating dataset

First you need to download simple wiki dump from link. For the purposes of our project we used the dump from 20th of February 2024.

```bash
curl -o .dump/wiki-dump.xml.bz2 https://dumps.wikimedia.org/simplewiki/20240220/simplewiki-20240220-pages-articles.xml.bz2
bzip2 -d .dump/wiki-dump.xml.bz2
```

Then you can use `bin/create_dataset.py` script to generate dataset from the dump. The script will generate a dataset in the form of a JSON file with the following fields: `title`, `text`, `categories`. The `categories` column is a list of categories that the article belongs to. For the purpose of our project we used only articles that belong to the following categories: `Diseases by causing agent`.

```bash
python bin/create_dataset.py --input_path='./.dump/wiki-dump.xml' --output_path='./dataset/diseases.json' --parent_category='Diseases by causing agent'
```

To split the dataset use:

```bash
python create_split.py --dataset_dir='dataset/diseases.json' --save_dir='dataset_split'
```

To eval model use:

```bash
python3 bin/eval_metrics.py --model_name='dumb' --dataset_name='diseases' --encoder_name='MiniLM'
```

To create your own model please inherit from abstract class in `abstract_model.py`.
To eval your own model / dataset / encoder please add it in `utils/common_utils.py`.
