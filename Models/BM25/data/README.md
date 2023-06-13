## Directory Overview 

This folder must contain the legal corpus in the `corpus/` subdirectory when running `run_script.py`. You can obtain the dataset via request, as mentioned in the project [README.md](../../../README.md).

A valid corpus contains a `query/`, `candidate/` directory containing current and prior cases respectively. Also a `label.json` file is required which contains the ground truth labels for evaluating F1 scores.
    
Additionally, the `data` folder contains the following auxiliary scripts : 
1. `remove_citation_sentences.py` : Takes a corpus in the standard dataset format of query/, candidate/, labels.json and removes sentences containing the `<CITATION>` keyword (please see the paper for details). Used to create the citation-sentence removed IL-PCR corpus.
2. `make_events_corpus.ipynb` : Creates the events, atomic and iouf corpuses from segment dictionaries in `segment_dictionaries`. The segment dictionary contains corpus events obtained from the event extraction pipeline. Please see `<Fill Here, reference to other folders readme.md>` for details. Note for first time users : The segment dictionaries must be produced by the event extraction pipeline and the results should be present in the `segment_dictionaries` directory.
3. `make_RR_corpus.ipynb` : Creates the RR corpus from segment dictionaries present in `segment_dictionaries/RR` directory.
4. `convert_labels_file.ipynb` : Converts `.csv` ground truth files from the COLIEE dataset into the more convenient `.json` format that we use.
