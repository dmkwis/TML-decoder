# Unveiling the Semantic Essence: Text Embeddings Summarization

## Introduction

The project delves into the problem of generating concise and accurate descriptions for clusters of documents represented as text embeddings. Text embeddings are high-dimensional vector representations of text that capture semantic meaning. By summarizing these clusters effectively, we aim to unlock valuable insights from large text collections, enhance automated summarization capabilities, and improve machine understanding of language.

## Problem Definition

Given a set of documents ${D_1, D_2, ..., D_n}$, each represented as a vector $v_i$ in a high-dimensional space, and known clusters ${C_1, C_2, ..., C_k}$ of documents, the objective is to generate a description $S_j$ for each cluster $C_j$. The generated description should maximize the cosine similarity between its vector representation ($s_j$) and the centroid of the cluster ($μ_ j$).

## Aplications and Significance

The ability to generate meaningful summaries of text embedding clusters has several important applications:

1. **Vector Databases:** Quickly understanding the thematic essence of large text collections without needing to access the original documents.
2. **Automated Summarization at Scale:** Contributing to the field of automated text summarization by providing a scalable method to condense and interpret vast amounts of text data.
3. **Enhancing Machine Understanding:** Aiding in the debugging of encoder models that are widely used in natural language processing tasks.

## New Dataset

To tackle this problem, a new dataset was created consisting of:

- 2200 abstracts with 20,000 keywords sourced from the paper [_Improved automatic keyword extraction given more linguistic knowledge_](https://aclanthology.org/W03-1028.pdf).
- 595 clusters containing at least 5 documents each.
- Summaries of documents tagged with specific keywords, generated using GPT-4. These summaries serve as a benchmark for evaluating the performance of our methods.

## Proposed Methodology

### Hard Prompting

This approach involves adding initial prompts to guide the language model at the beginning of the generation process. Prompts like _"These documents describe..."_ or _"Overview of this topic is..."_ are used to steer the model towards producing relevant summaries.

### Soft Prompting

Soft prompting, inspired by the work of Liu et al. (2022), involves two techniques:

1. **Choosing new nodes with soft prompts:** Instead of randomly selecting nodes during the generation process, soft prompts are used to guide the selection, potentially leading to more coherent and relevant summaries.
2. **Creating new nodes with soft prompts:** To address the issue of language models generating words based on probability distributions that may not align with the target domain, soft prompts are used to create new states that are closer to the cluster's average embedding. This helps ensure that the generated summaries are more semantically aligned with the cluster's content.

## Results and Evaluation

<!-- TODO -->

Tu jest jeszcze pusto, bo nie mamy wszystkiego.
