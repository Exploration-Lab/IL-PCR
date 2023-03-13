## Directory Overview

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

## Auxiliary scripts
1. `replicate_configs.py` : Each config directory contains information for experiments conducted on the same corpus but with different `n_gram` value. Therefore, it is useful to take a single config file and programmatically create the remaining config files by changing the `n_gram` values. Example usage is 
    ```bash 
    python3 ./replicate_configs.py path/to/config_folder
    ```

2. `flip_configs.py` : Our test and train corpus names differ in only the test/train identifier, for e.g. `ik_test_iouf`, `ik_train_iouf`. Hence, a config directory for the test corpus can be converted to the config directory for the train corpus. Example usage is 
    ```bash 
    python3 ./flip_configs.py path/to/config_folder
    ```
