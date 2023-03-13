# U-CREAT: Unsupervised Case Retrieval using Events extrAcTion

This folder contains the official code for all word based transformer experiments conducted in the paper `U-CREAT: Unsupervised Case Retrieval using Events extrAcTion`. The experiments include `Segment-Doc Transformer(full document)` and `Transformer (top 512 tokens)`.

## Project Overview
1. `transformer_score.py`

    The main python script which runs the transformer experiments. Takes as input a config file containing experiment details and saves the experiment output/metrics in the `exp_results` folder with a time stamp of when the experiment was been started. The formats of the input (config files) and the experiment output are described below.

2. `config_files/`

   This folder contains several sample input config files for `transformer_score.py`. Config files are included for the models `bert-base-uncased`, `bert-finetuned`, `distilbert-base-uncased`, `distilbert-finetuned`, `InLegalBERT`, `InCaseLawBERT` being evaluated on the corpuses `COLIEE21`, `IL-PCR`, `citation-sentence removed IL-PCR`.
   Each config file must contain the following entries for the run script to complete the experiment : 

   1. `path_prior_cases` : Path to the folder containing prior/candidate cases. The cases must be present as `.txt` files inside the folder.
   2. `path_current_cases` : Path to the folder containing current/query cases. The cases must be present as `.txt` files inside the folder.
   3. `true_labels_json` : Path to the label.json file containing the ground truth labels for candidate cases considered relevant per query case. This is required to compute F1@K and related evaluation metrics. The general format for this json file is  
   ```json
    {
        "Query Set": [  // contains separate dictionary for each query case 
            {
                "id" : <string, case name>, 
                "query_name": <string, case name>
                "relevant candidates": [    // contains ids of relevant cases
                        ...
                ]
            }, ...
        ], 
        "Candidate Set": [
            {
                "id": <string, case name>
            }, ...
        ]
    }
   ```
   Please see an example label.json file included with the dataset for a concrete example.

   4. `checkpoint` : Checkpoint of the model to be used to produce the vector embeddings of documents.

   5. `top512` : Indicates whether to use the top 512 tokens of a document to obtain its vector representation. If set to `True` the top 512 tokens are taken. If set to `False` the document is divided into chunks of 512 tokens and the vector representation of each chunk is concatentated to produce the final vector embedding. Please see the paper for details.

3. `spawn_transformers.sh`

    This is a helper script which spawns experiments with different config files at once. Our config files are arranged folder-wise and are named `config_1, config_2, ...` and so on. This allows you to automate running the experiments in the paper.

4. `data`

    This folder must contain the legal corpus in the `corpus/` subdirectory when running `transformer_score.py`. A valid corpus contains a `query/`, `candidate/` directory containing current and prior cases respectively. Also a `label.json` file is required which contains the ground truth labels for evaluating F1 scores.

5. `evaluate_at_K.py`

    Evaluates the micro F1 score between a ground truth file (`label.json`) and a similarity score csv produced by `transformer_score.py`. The similarity score csv contains a relevance score for each query X candidate pair. Please check the paper for more details on how the results are evaluated.

6. `get_exp_results.py`

    This script fetches the experiment results (recall, precision and F1 at K) as reported in the paper. The value of $K$ is determined using the results for the trainset of ILPCR (please see the paper for details). Hence, the results over any test corpus requires running `transformer_score.py` on the counter-part train corpus first. Once both results are available the script fetches the best $K$ obtained from the train corpus and reports the appropriate F1@K value on the test set.

7. `model_checkpoints/`

    Contains model checkpoints for finetuned bert and distilbert for the experiments in the paper. If you require these models please contact the authors.

8. `transformer_score.ipynb`

    A `.ipynb` format of `transformer_score.py`, useful for visualization, interactive coding and educational purposes.

## Installation
1. pip requirements are listed in the `requirements.txt` file. Install using 
    ```bash
    python3 -m pip install -r requirements.txt
    ```

## Usage
1. `transformer_score.py` : Requires a config file to run and a dataset to be present at `data/corpus/` in the standard dataset format. Example usage is 
    ```
    python3 ./transformer_score.py path/to/config_file.json
    ```

2. `spawn_transformers.sh` : Multiple experiments can be run simultaneously with running python files in the background and redirecting their standard output/error streams. An example is :
    ```bash
    python3 -u ./transformer_score.py config_1_path 1>./logs/log1 2>&1 & 
    python3 -u ./transformer_score.py config_2_path 1>./logs/log2 2>&1 & 
    ```
    Our `config_files/` folder contains arranged configs of each word-based experiment in the paper. The results in the paper can be recreated by uncommenting the line for the required experiment and simply using 
    ```bash
    ./spawn_transformers.sh
    ```

3. `get_exp_results.py` : Obtains the recall, precision and F1 @ K values by fetching experiment statistics saved in `exp_results/`. Results to be displayed can be changed by modifying the file internally (no command line option for that). Example usage is 
    ```bash
    python3 ./get_exp_results.py
    ```
