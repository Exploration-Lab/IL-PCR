# U-CREAT: Unsupervised Case Retrieval using Events extrAcTion

This folder contains the official code for generating the processed data using the raw data and also the code for SBERT experiments.

# Following modules are explained in this ReadMe.

1. Extract Events

   1.1 Event Extraction from the dataset.

2. IOU

   2.1 Similarity score calculation between queries and candidates using the events extracted

   2.2 Creation of Matching Events (IOU) filtered data corpus. This processed corpus is further
   used by BM25 module.

3. SBERT

   3.1 Similarity score calculation between queries and documents using pretrained and
   finetuned BERT and distilRoBERTa models. (Checkpoints for the best performing models are provided).

# Requirements:

1.  Libraries:

    1.1 Install spaCy: Follow https://spacy.io/usage and install the "en_core_web_trf" model, requires GPU.

    1.2 Install transformers: pip install transformers

    1.3 Install sentence transformers: pip install sentence-transformers

2.  Folder structure:

    - UCREAT
      - data (This folder contains all the raw and processed datasets)
        - \<dataset> (folder with the name of your dataset Ex: ilpcr, coliee etc)
          - \<split_type> (folder with split_type Ex: test, train, dev)
            - \<files_type> (folder with files_type Ex: query, candidate)
      - Events_Extraction
        - "extract_events.py"
        - "input_details.json"
      - IOU
        - "Event_IOU_Sim.py" (Calculates similarity scores using events)
        - "Event_IOU_Sim_input_details.json" (Give input here for the above py file)
        - "IOU_filtered_text_dict.py" (Creates the Matching Events (IOU) filtered data corpus)
        - "IOU_filtered_input_details.json" (Give input here for the above py file)
      - RR
        - Refer to the github : https://github.com/Exploration-Lab/Rhetorical-Roles
        - Replace this folder with the RR folder given in the above github.
      - SBERT
        - output (This folder contained the finetuned models of the SBERT)
          - c21
            - bert_base
              - 48103
              - 91833
            - distilroberta_base
              - 12023
              - 22953
          - ilpcr
            - bert_base
              - 81939
              - 156429
            - distil_roberta
              - 20482
              - 39102
        - "SBERT_Sim.py" (Similarity Score calculation using SBERT)
        - "SBERT_input_details.json" (Give inputs here for SBERT_Sim.py file)

    Note:

    c21 - finetuned models for coliee21

    ilpcr - finetuned models for ilpcr

# Detailed Explanation to use the above modules.

# 1. Events_Extraction

Use this module for event extraction from the dataset.
Follow the below steps to run the "extract_events.py" file.

1.  Give input using "input_details.json" file
2.  Example for the input is shown below:

        {
            "input_root": "./data",
            "dataset": "ilpcr",
            "split_type": "train",
            "files_type": "query",
            "output_root": "./data/ilpcr_processed"
        }

    Note:

    - "input_root" is the base directory consisting of all the data.
    - "dataset" - folder in the input_root directory (Ex: ilpcr, coliee etc)
    - "split_type" - folder inside dataset folder (Ex: test, train, dev)
    - "files_type" - folder inside split_type folder (Ex: query, candidate)
    - "output_root" - Give full path to save the processed data (Preferred: "./data/dataset_processed")

3.  This file "extract_events.py" requires GPU. Using the following command (without quotes) to run the file.
    "CUDA_VISIBLE_DEVICES=x python extract_events.py" (where x is GPU number. Ex: 0,1,2,3)
4.  Folders and other processed data files created by this module.

    4.1 All the processed information is saved in the ./data folder.

    The following is its structure.

    (Showing only newly created folders and files inside data folder)

    - data
      - \<dataset>\_processed
        - \<dataset>
          - \<split_type>
            - \<files_type>
              - "file_name.json" (all processed files are stored as jsons)
            - "event_doc_line_text\_\<split_type>\_\<dataset>\_\<files_type>.pkl"
            - "segment_dictionary\_\<split_type>\_\<dataset>\_\<files_type>.sav"
            - "sent_data\_<split_type>\_\<dataset>\_\<files_type>.sav"

    Example: "dataset" = ilpcr, "split_type" = test, "files_type" = candidate

    Starting from ilpcr_processed folder, everything is created by the "extract_events.py"

    - data
      - ilpcr_processed
        - ilpcr
          - test
            - candidate
              - "file_name.json" (all processed files are stored as jsons)
            - "event_doc_line_text_test_ilpcr_candidate.pkl"
            - "segment_dictionary_test_ilpcr_candidate.sav"
            - "sent_data_test_ilpcr_candidate.sav"

# 2. RR

- Refer to the github : https://github.com/Exploration-Lab/Rhetorical-Roles
- Replace the RR folder with the folder given in the above github.

# 3. IOU

Use this module for Jaccard (IOU) similarity score calculation and Matching Events (IOU) filtered data corpus generation from the dataset.

Follow the below steps to run the "Event_IOU_Sim.py" file.

1.  Give input using "Event_IOU_Sim_input_details.json" file
2.  Example for the input is shown below:

        {
            "dataset": "ilpcr",
            "split_type": "test",
            "query_segment_dictionary_path": "../data/ilpcr_processed/ilpcr/test/segment_dictionary_test_ilpcr_query.sav",
            "candidate_segment_dictionary_path": "../data/ilpcr_processed/ilpcr/test/segment_dictionary_test_ilpcr_candidate.sav"
        }

# Note:

- "dataset" - folder in the input_root directory (Ex: ilpcr, coliee etc)
- "split_type" - folder inside dataset folder (Ex: test, train, dev)
- "query_segment_dictionary_path" - Give full path to the query segment dictionary
- "candidate_segment_dictionary_path" - Give full path to the candidate segment dictionary
  (Preferred: "../data/ilpcr_processed/ilpcr/test/segment_dictionary\_\<split_type>\_\<dataset>\_candidate.sav")

3. Folders and other processed datafiles created by this module.

   3.1 All the processed information is saved in the ./Sim_CSVs folder in the current location (/IOU/). The following is its structure.

   (Showing only newly created folders and files inside IOU folder)

   - Sim_CSVs
     - dataset
       - \<dataset>\_\<split_type>\_IOU_sim.csv"

   Example: "dataset" = ilpcr, "split_type" = test

   Starting from Sim_CSVs folder, everything is created by the "Event_IOU_Sim.py"

   - Sim_CSVs
     - ilpcr
     - "ilpcr_test_IOU_sim.csv"

Follow the below steps to run the "IOU_filtered_text_dict.py" file.

1.  Give input using "IOU_filtered_input_details.json" file
2.  Example for the input is shown below:

        {
            "dataset": "ilpcr",
            "split_type": "test",
            "output_dir": "../data/ilpcr_processed",
            "seg_data_dir_path": "../data/ilpcr_processed",
            "event_doc_line_text_dir_path": "../data/ilpcr_processed"
        }

# Note:

- "dataset" - folder in the input_root directory (Ex: ilpcr, coliee etc)
- "split_type" - folder inside dataset folder (Ex: test, train, dev)
- "output_dir" - Give full path to the query segment dictionary
- "seg_data_dir_path" - Give full path to the segment directories.(both query and candidate should be present in this location)
- "event_doc_line_text_dir_path" - Give full path to the event_doc_line_text pickle files.(both query and candidate should be present in this location. Preferred: "../data/ilpcr_processed/)

From the above locations, the script will load segment dictionaries and event doc line text pickle files for both query and candidates.

Path from which these are loaded is as follows:

Make sure that the above mentioned files are present in these locations.

1. seg_path_candi = seg_data_dir_path+"/"+\<dataset>+"/"+\<split_type> + "segment_dictionary\_"  
   +\<split_type>+"\_"+\<dataset>+"\_candidate.sav"
2. seg_path_query = seg_data_dir_path+"/"+\<dataset>+"/"+\<split_type> + "/segment_dictionary\_"
   +\<split_type>+"\_"+\<dataset>+"\_query.sav"
3. event_doc_line_text_path_candi = event_doc_line_text_dir_path+"/"+\<dataset>+"/"
   +\<split_type>+ "/event_doc_line_text\_"+\<split_type>+"\_"+\<dataset>+"\_candidate.pkl"
4. event_doc_line_text_path_query = event_doc_line_text_dir_path+"/"+\<dataset>+"/"
   +\<split_type>+ "/event_doc_line_text\_"+\<split_type>+"\_"+\<dataset>+"\_query.pkl"

5. Folders and other processed datafiles created by this module.

   3.1 Processed information is saved in ../data/\<dataset>\_processed/\<dataset>/\<split_type>
   folder.

   The following is the structure.

   - data
     - \<dataset>\_processed
       - \<dataset>
         - \<test>
           - "IOU_filtered_text_dict\_\<dataset>\_\<split_type>.sav"

   Example: "dataset" = ilpcr, "split_type" = test

   - data
     - ilpcr_processed
       - ilpcr
         - test
           - "IOU_filtered_text_dict_ilpcr_test.sav"

# 4. SBERT

Use this module for SBERT similarity score calculation.

Follow the below steps to run the "SBERT_Sim.py" file.

1.  Give input using "SBERT_input_details.json" file
2.  Example for the input is shown below:

        {
            "dataset": "ilpcr4",
            "split_type": "train",
            "data_dir_path": "../data",
            "query_segment_dictionary_path": "../data/ilpcr_processed/ilpcr/train/sent_data_train_ilpcr_query.sav",
            "candidate_segment_dictionary_path": "../data/ilpcr_processed/ilpcr/train/sent_data_train_ilpcr_candidate.sav",
            "model_type": "bert",
            "model_id": 156960,
            "model_name": "bert-base-uncased",
            "model_path": "./output/ilpcr4/bert_base/156960/"
        }

# Note:

- "dataset" - folder in the input_root directory (Ex: ilpcr, coliee etc)
- "split_type" - folder inside dataset folder (Ex: test, train, dev)
- "data_dir_path" - path to data directory
- "query_segment_dictionary_path" - path to query segment dictionary
- "candidate_segment_dictionary_path" - path to candidate segment dictionary
- "model_type" - give the type of model (Ex: bert, distilroberta)
- "model_id" - give model id (0 for pretrained, ddddd - for finetuned). Finetuned models are stored in these ids.
- "model_name" - Give pretrained model name (Ex: "bert-base-uncased", "distilroberta-base")
- "model_path": - Give path for the model chosen above.
  (Preferred:"./output/ilpcr4/bert_base/156960/")

Models are stored in the "output/dataset/model/model_id" path

3. Folders and other processed datafiles created by this module.

   3.1 All the processed Sim information is saved in the ./Sim_CSVs folder in the current location (/SBERT/). The following is its structure.

   (Showing only newly created folders and files inside IOU folder)

   - Sim_CSVs
     - dataset
       - \<model_type>\_base\_\<dataset>\_\<split_type>\_\<model_type>\_\<model_id>.csv"

   Example: "dataset" = ilpcr, "split_type" = test

   Starting from Sim_CSVs folder, everything is created by the "Event_IOU_Sim.py"

   - Sim_CSVs
     - ilpcr
       - bert_base
     - "ilpcr_train_bert_0.csv"
