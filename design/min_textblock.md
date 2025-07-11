# Specification for Tiny-Text-Block

This document outlines the minimal transform pipeline for processing the sentiment analysis CSV data. It replaces the default `fastai.TextBlock` with a simpler, custom set of transforms.

## Core Transform Pipeline

The pipeline processes one row of the DataFrame at a time.

| Stage # | Transform            | Purpose                                            | Input Type       | Output Type      |
|:--------|:---------------------|:---------------------------------------------------|:-----------------|:-----------------|
| 1       | **ColReader('text')**| Reads the raw review text from a DataFrame row.    | `pd.Series`      | `str`            |
| 2       | **MiniTokenizer**    | Splits string into list of cleaned, lower-cased tokens. | `str`            | `List[str]`      |
| 3       | **Numericalize**     | Converts each token into a unique integer ID.      | `List[str]`      | `TensorText`     |

---

### Component Details

*   **ColReader:** A standard fastai transform. No custom implementation needed.
*   **MiniTokenizer:** A custom Python function/transform we will build in Step 3. It will be very simple: lowercase, remove punctuation, and split by whitespace.
*   **Numericalize:** A standard fastai transform. It will build its vocabulary from the training data.