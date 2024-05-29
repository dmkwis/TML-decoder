"""
This script generates a dataset from the Inspec dataset. The dataset consists of a list of documents, a list of keywords
associated with each document, and a summary of the documents associated with each keyword.

The script reads the documents and keywords from the Inspec dataset, generates summaries for the documents associated
with each keyword, and saves the dataset to the specified output directory. Documents used for generating summaries are
sampled using `seed` and `sample_size` parameters.
For the keywords that have only one document associated with them, the script does not generate summaries. We ignore
these keywords because the summaries generated from a single document are not reliable.

The script uses the OpenAI API to generate summaries. You need to have an OpenAI API key set in the `OPENAI_API_KEY`
environment variable to use this script.

Usage:
    python bin/create_inspec_dataset.py --base_path .dump/Inspec --sample_size 3 --seed 42

Arguments:
    base_path: The base path where the Inspec dataset is located.
    sample_size: The number of documents to sample for generating summaries.
    seed: The seed to use for sampling documents.

Output:
    The script saves the dataset to the specified output directory. The dataset consists of a JSONL file containing
    the documents, keywords, and summaries, and a CSV file containing the summaries for each keyword with the
    documents it was generated from.

    The output path is `dataset/inspec/dataset.jsonl` and `dataset/inspec/summaries.csv`.
"""

import json
import os
import random
import re
from typing import Dict, List

import fire
from openai import OpenAI
import pandas as pd
from tqdm import tqdm


def _generate_summary(client: OpenAI, docs: List[str], keyword: str) -> str:
    """
    Generate a summary from the list of documents

    :param keywords: The keywords of the article
    :returns: The summary of the article
    """
    joined_docs = "\n".join(docs)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                You have a collection of scientific articles that are tagged with a keyword inside [] brackets, each article is separated by a new line.
                Generate a concise summary that encompasses the key insights from all the articles collectively and includes keyword that is inside [] brackets.
                Your summary should capture the overarching themes, main findings, and notable points discussed across the articles.
                Focus on summary of all of them at once, not individual articles. Write a summary that is only a couple of words long.
                Respond ONLY with the summary of the articles. Remember to be very concise and capture the key insights. It has to be descriptive and human friendly.
                Please note that the provided articles are only a sample of the entire collection that are related to single keyword that is inside [] brackets at the beginning of the message.
                """,
            },
            {"role": "user", "content": f"[{keyword}]:\n{joined_docs}"},
        ],

    )

    response = response.choices[0].message.content

    with open(".dump/debug.log.jsonl", "a") as f:
        debug = {
            "keyword": keyword,
            "input": joined_docs,
            "output": response,
            "is_error": len(response.split(".")) > 2 if response else True,
        }
        debug = json.dumps(debug)
        f.write(debug + "\n")

    return response if response else ""


def _clean_doc(doc: str) -> str:
    doc = re.sub(r"\W+", " ", doc)
    doc = re.sub(r"\s+", " ", doc)

    return doc


def _read_docs(base_path: str) -> List[str]:
    path = f"{base_path}/docsutf8"
    files = os.listdir(path)

    docs: List[str] = []
    for file in files:
        with open(f"{path}/{file}", "r") as f:
            doc = _clean_doc(f.read())
            docs.append(doc)

    return docs


def _clean_keyword(keyword: str) -> str:
    keyword = keyword.replace("\t", " ").replace("\n", " ").replace("\r", " ").replace("\f", " ").replace("\v", " ").strip().lower()
    keyword = re.sub(r"\s+", " ", keyword)
    return keyword


def _read_keywords(base_path: str) -> List[List[str]]:
    """
    Read keywords from files in the specified base path.

    :param base_path: The base path where the keyword files are located.

    :returns: A list of lists, where each inner list contains the keywords from a file.
    """

    files = os.listdir(base_path + "/keys")

    keywords: List[List[str]] = []

    for file in files:
        with open(base_path + "/keys/" + file, "r") as f:
            keywords_list = f.read().splitlines()
            keywords_list = [_clean_keyword(keyword) for keyword in keywords_list]
            keywords_list = list(set(keywords_list))
            keywords_list.sort()

            keywords.append(keywords_list)

    return keywords


from typing import List


def _get_keyword_occurences(keywords: List[List[str]]) -> Dict[str, List[int]]:
    """
    Get the occurrences of keywords in a list of lists.

    :param keywords: A list of lists containing keywords.

    :returns: A dictionary where the keys are keywords and the values are lists of indices
              indicating the occurrences of the keywords in the input list of lists.
    """
    keyword_occurences = {}

    for i in range(len(keywords)):
        for keyword in keywords[i]:
            if keyword in keyword_occurences:
                keyword_occurences[keyword].append(i)
            else:
                keyword_occurences[keyword] = [i]

    return keyword_occurences


def main(base_path: str = ".dump/Inspec", sample_size: int = 3, seed: int = 42):
    random.seed(seed)
    client = OpenAI()

    docs = _read_docs(base_path)
    with open(".dump/debug.log", "w") as f:
        f.write(docs[0])
    keywords = _read_keywords(base_path)

    keyword_occurences = _get_keyword_occurences(keywords)
    keywords_descriptions: Dict[str, str] = {}
    keyword_docs_sub: Dict[str, List[str]] = {}

    for keyword in tqdm(keyword_occurences.keys()):
        docs_sub = [docs[i] for i in keyword_occurences[keyword]]

        if len(docs_sub) == 1:
            continue

        sample = random.sample(docs_sub, min(sample_size, len(docs_sub)))

        keywords_descriptions[keyword] = _generate_summary(client, sample, keyword)
        keyword_docs_sub[keyword] = sample

    existing_keywords = list(keywords_descriptions.keys())
    filtered_keywords = [[keyword for keyword in keywords_list if keyword in existing_keywords] for keywords_list in keywords]

    summaries_df = pd.DataFrame(
        {
            "keyword": list(keywords_descriptions.keys()),
            "description": list(keywords_descriptions.values()),
            "subs": list(keyword_docs_sub.values()),
        }
    )
    summaries_df.to_csv("dataset/inspec/summaries.csv", index=False)

    dataset_df = pd.DataFrame(
        {
            "doc": docs,
            "keywords": filtered_keywords,
            "summary": [[keywords_descriptions[keyword] for keyword in keywords_list] for keywords_list in filtered_keywords],
        }
    )
    dataset_df = dataset_df[dataset_df["summary"].apply(lambda x: len(x) > 0)]

    dataset_df.to_json("dataset/inspec/dataset.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    fire.Fire(main)
