# RAG Pipeline for Human Nutrition Textbook

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) pipeline. The RAG model downloads a human nutrition textbook, retrieves context from the textbook, and generates relevant responses. The implementation uses sentence embeddings created with `all-mpnet-base-v2` and leverages the `unsloth/Phi-3-mini-4k-instruct` language model.

## Introduction

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline, a method that combines retrieval of relevant documents with generative language models to provide context-aware responses. The RAG model first downloads a human nutrition textbook, processes it to create sentence embeddings, and then retrieves relevant contexts to generate answers to user queries.

## Installation

To get started with this project, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/yourusername/rag-nutrition-pipeline.git
cd rag-nutrition-pipeline
pip install -r requirements.txt
```

## Usage

To run the RAG pipeline, open the Jupyter notebook `RAG_notebook.ipynb` and follow the steps outlined in the notebook. The notebook includes code to:

1. Download the human nutrition textbook.
2. Process the textbook to create sentence embeddings using the `all-mpnet-base-v2` model.
3. Retrieve relevant contexts from the textbook.
4. Generate responses using the `unsloth/Phi-3-mini-4k-instruct` language model.

## Requirements

The project dependencies are listed in the `requirements.txt` file. 


Ensure you have all the required packages by installing them via `pip`:

```bash
pip install -r requirements.txt
```

## Project Structure

The repository contains the following files:

- `requirements.txt`: List of dependencies required for the project.
- `RAG_notebook.ipynb`: Jupyter notebook with the complete implementation of the RAG pipeline.

## Acknowledgments

- The sentence embeddings are created using the `all-mpnet-base-v2` model from the `sentence-transformers` library.
- The language model used for generation is `unsloth/Phi-3-mini-4k-instruct`.

This project is inspired by the need to create an efficient and accurate context-aware generative model for educational purposes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

