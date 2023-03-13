# U-CREAT: Unsupervised Case Retrieval using Events extrAcTion

This folder contains the official code for all word-based BM25 experiments conducted in the paper `U-CREAT: Unsupervised Case Retrieval using Events extrAcTion`. The experiments include `word-based BM25`, `non-atomic events`, `atomic events`, `events filtered docs` and `RR filtered docs`.

## Project Overview
1. `run_script.py`

    The main python script which runs the BM25 experiment. Takes as input a config file containing experiment details and saves the experiment output/metrics in the `exp_results` folder with a time stamp of when the experiment was been started. The formats of the input (config files) and the experiment output are described below.

2. `config_files/`

   This folder contains several sample input config files for `run_script.py`. Config files are included for the COLIEE21, ILPCR and citation-sentence removed ILPCR corpus. The suffixes `events`, `atomic`, `iouf` and `RR` indicate config files for the experiments `non-atomic events`, `atomic events`, `events filtered docs` and `RR filtered docs` in the paper respectively.
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

   4. `n_gram` : The gram value for which to run BM25. Unigram, bigram, etc.

3. `spawn_process.sh`

    This is a helper script which spawns experiments with different config files at once. Our config files are arranged folder-wise and are named `config_1, config_2, ...` and so on. This allows you to automate running the experiments in the paper.

4. `data`

    This folder must contain the legal corpus in the `corpus/` subdirectory when running `run_script.py`. A valid corpus contains a `query/`, `candidate/` directory containing current and prior cases respectively. Also a `label.json` file is required which contains the ground truth labels for evaluating F1 scores.
    
    Additionally, the `data` folder contains the following auxiliary scripts : 
    1. `remove_citation_sentences.py` : Takes a corpus in the standard dataset format of query/, candidate/, labels.json and removes sentences containing the `<CITATION>` keyword (please see the paper for details). Used to create the citation-sentence removed IL-PCR corpus.
    2. `make_events_corpus.ipynb` : Creates the events, atomic and iouf corpuses from segment dictionaries present in `segment_dictionaries`. The segment dictionary contains corpus events obtained from the event extraction pipeline. Please see `<Fill Here, reference to other folders readme.md>` for details.
    3. `make_RR_corpus.ipynb` : Creates the RR corpus from segment dictionaries present in `segment_dictionaries/RR/` directory.

5. `evaluate_at_K.py`

    Evaluates the micro F1 score between a ground truth file (`label.json`) and a similarity score csv produced by `run_script.py`. The similarity score csv contains a relevance score for each query X candidate pair. Please check the paper for more details.

6. `get_exp_results.py`

    This script fetches the experiment results (recall, precision and F1 at K) as reported in the paper. The value of $K$ is determined using the results for the trainset of ILPCR (please see the paper for details). Hence, the results over any test corpus requires running `run_script.py` on the counter-part train corpus first. Once both results are available the script fetches the best $K$ obtained from the train corpus and reports the appropriate F1@K value on the test set.

7. `run_script.ipynb`

    A `.ipynb` format of `run_script.py`, useful for visualization, interactive coding and educational purposes.

## Installation
1. pip requirements are listed in the `requirements.txt` file. Install using 
    ```bash
    python3 -m pip install -r requirements.txt
    ```

## Usage
1. `run_script.py` : Requires a config file to run and a dataset to be present at `data/corpus/` in the standard dataset format. Example usage is 
    ```
    python3 ./run_script.py path/to/config_file.json
    ```

2. `spawn_process.sh` : Multiple experiments can be run simultaneously with running python files in the background and redirecting their standard output/error streams. An example is :
    ```bash
    python3 -u ./run_script.py config_1_path 1>./logs/log1 2>&1 & 
    python3 -u ./run_script.py config_2_path 1>./logs/log2 2>&1 & 
    ```
    Our `config_files/` folder contains arranged configs of each word-based experiment in the paper. After all of the corpuses have been obtained/created from segment dictionaries the experiments can be run by simply using 
    ```bash
    ./spawn_process.sh
    ```

3. `evaluate_at_K.py` : Takes as input path to a ground truth json file and the similarity score csv. Example usage is 
    ```bash 
    python3 ./evaluate_at_k.py --ground-truth path/to/labels.json --sim-csv path/to/sim.csv
    ```

4. `get_exp_results.py` : Obtains the recall, precision and F1 @ K values by fetching experiment statistics saved in `exp_results/`. Results to be displayed can be changed by modifying the file internally (no command line option for that). Example usage is 
    ```bash
    python3 ./get_exp_results.py | sort -k 1 -k 3 # sort by experiment name and n_gram value
    ```
