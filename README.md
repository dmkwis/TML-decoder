# tml-decoder

Team machine learning project for University of Warsaw in collaboration with Huawei - Generative decoder for sentence embeddings.

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
