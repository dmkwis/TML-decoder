"""
This script is used to create a dataset from a wikipedia xml file. The dataset
will contain the articles and their respective categories. The categories are
found by searching for the parent category and then finding all the subcategories
for the parent category. The articles are then found for each subcategory.

The script can be run from the command line using the following command:
python bin/create_dataset.py --input_path <input_path> --output_path <output_path> --parent_category <parent_category>

Where:
- input_path: The location of the xml file
- output_path: The location of the output file
- parent_category: The parent category to find the articles for
"""

import xml.etree.ElementTree as ET
import codecs
import re

import pandas as pd
from loguru import logger
import fire


def clean_article(article_txt: str) -> str:
    """
    Clean the article text by removing all the unnecessary information and formatting
    that is used by wikipedia and is not useful for our purposes.

    :param article_txt: The article text to be cleaned
    :returns: The cleaned article text
    """

    article_txt = article_txt[: article_txt.find("==")]
    article_txt = re.sub(r"{{.*}}", "", article_txt)
    article_txt = re.sub(r"\[\[File:.*\]\]", "", article_txt)
    article_txt = re.sub(r"\[\[Image:.*\]\]", "", article_txt)
    article_txt = re.sub(r"\n: \'\'.*", "", article_txt)
    article_txt = re.sub(r"\n!.*", "", article_txt)
    article_txt = re.sub(r"^:\'\'.*", "", article_txt)
    article_txt = re.sub(r"&nbsp", "", article_txt)
    article_txt = re.sub(r"http\S+", "", article_txt)
    article_txt = re.sub(r"\d+", "", article_txt)
    article_txt = re.sub(r"\(.*\)", "", article_txt)
    article_txt = re.sub(r"Category:.*", "", article_txt)
    article_txt = re.sub(r"\| .*", "", article_txt)
    article_txt = re.sub(r"\n\|.*", "", article_txt)
    article_txt = re.sub(r"\n \|.*", "", article_txt)
    article_txt = re.sub(r".* \|\n", "", article_txt)
    article_txt = re.sub(r".*\|\n", "", article_txt)
    article_txt = re.sub(r"{{Infobox.*", "", article_txt)
    article_txt = re.sub(r"{{infobox.*", "", article_txt)
    article_txt = re.sub(r"{{taxobox.*", "", article_txt)
    article_txt = re.sub(r"{{Taxobox.*", "", article_txt)
    article_txt = re.sub(r"{{ Infobox.*", "", article_txt)
    article_txt = re.sub(r"{{ infobox.*", "", article_txt)
    article_txt = re.sub(r"{{ taxobox.*", "", article_txt)
    article_txt = re.sub(r"{{ Taxobox.*", "", article_txt)
    article_txt = re.sub(r"\* .*", "", article_txt)
    article_txt = re.sub(r"<.*>", "", article_txt)
    article_txt = re.sub(r"\n", "", article_txt)
    article_txt = re.sub(
        r"\!|\"|\#|\$|\%|\&|\'|\(|\)|\*|\+|\-|\/|\:|\;|\<|\=|\>|\?|\@|\\|\^|\_|\`|\{|\||\}|\~",
        " ",
        article_txt,
    )
    article_txt = re.sub(r"\[|\]", "", article_txt)
    article_txt = re.sub(
        r"\s*(\,|\.)\s*",
        lambda match: ". " if match.group(1) == "." else ", ",
        article_txt,
    )
    article_txt = article_txt.strip(" ")
    article_txt = re.sub(r" +", " ", article_txt)
    article_txt = article_txt.replace("\xa0", " ")
    return article_txt


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe by removing all the unnecessary information and formatting.

    :param df: The dataframe to be cleaned
    :returns: The cleaned dataframe
    """

    logger.info("Cleaning dataframe")
    df["article"] = df["article"].apply(clean_article)
    return df


def parse(file_location: str) -> pd.DataFrame:
    """
    Parse the xml file and return a dataframe with the articles and titles.

    :param file_location: The location of the xml file
    :returns: A dataframe with the articles and titles
    """

    logger.info(f"Reading file {file_location}")

    tree = ET.parse(file_location)
    root = tree.getroot()

    titles_list = []
    article_list = []

    for i, page in enumerate(
        root.findall("{http://www.mediawiki.org/xml/export-0.10/}page")
    ):
        title = ""
        for p in page:
            if p.tag == "{http://www.mediawiki.org/xml/export-0.10/}title":
                title = p.text
            if p.tag == "{http://www.mediawiki.org/xml/export-0.10/}revision":
                for x in p:
                    if x.tag == "{http://www.mediawiki.org/xml/export-0.10/}text":
                        article_txt = x.text

        if article_txt and title:
            article_list.append(article_txt)
            titles_list.append(title)

    logger.info(
        f"Successfully finished reading file. Found {len(article_list)} articles."
    )

    return pd.DataFrame((article_list, titles_list)).T.rename(
        columns={0: "article", 1: "title"}
    )


def find_categories(df: pd.DataFrame, parent_category: str) -> pd.Series:
    """
    Find the categories for the parent category.

    :param df: The dataframe with the articles and titles
    :param parent_category: The parent category to find the categories for
    :returns: A series with the categories for the parent category
    """

    logger.info(f"Finding categories for {parent_category}")
    rows = df[df["article"].str.contains(f"Category:{parent_category}")]
    result = rows["title"]

    logger.info(f"Found {len(result)} categories for {parent_category}")

    return result


def find_articles(df: pd.DataFrame, category: str):
    """
    Find the articles for the category.

    :param df: The dataframe with the articles and titles
    :param category: The category to find the articles for
    :returns: A series with the articles for the category
    """

    logger.info(f"Finding articles for {category}")

    result = df["article"].str.contains(f"{category}")
    result = result & ~df["title"].str.contains("Category:")

    logger.info(f"Found {result.sum()} articles for {category}")

    return result


def main(input_path: str, output_path: str, parent_category: str):
    raw_df = parse(input_path)

    df = raw_df.copy()

    categories = find_categories(raw_df, parent_category)

    df["category"] = pd.NA

    for category in categories:
        category = category.replace("Category:", "")
        articles = find_articles(raw_df, category)
        df.loc[articles, "category"] = category

    df = df.dropna(subset=["category"])

    logger.info(f"Found {len(df)} articles for {parent_category} category.")

    df = clean_df(df)

    logger.info(f"Writing to file {output_path}")
    df.to_json(output_path, orient="records", indent=2)


if __name__ == "__main__":
    fire.Fire(main)
