import os, sys, json, time, numpy as np
import evaluate_at_K
assert(len(sys.argv) == 2)  # pass the config file while running 

# save destination contains a time stamp followed by the process pid to ensure processes spawned together don't overwrite each other's folder
current_time = '_'.join(time.ctime().split()) + '_' + str(os.getpid())
save_folder = f'./exp_results/EXP_{current_time}'
os.makedirs(save_folder)

with open(sys.argv[1], 'r') as f:
    config_dict = json.load(f)
with open(save_folder + f'/config_file.json', 'w') as f:
    json.dump(config_dict, f, indent = 4)   # save the config file with which the experiments are run

path_prior_cases = config_dict['path_prior_cases']
path_current_cases = config_dict['path_current_cases']
true_labels_json = config_dict['true_labels_json']
n_gram = config_dict['n_gram']

assert(os.path.isdir(path_prior_cases))
assert(os.path.isdir(path_current_cases))
assert(os.path.isfile(true_labels_json))

bm25_results_save_dict_path = f'{save_folder}/bm25_results.sav'     # saves the dictionary containing each query X candidate score
filled_sim_csv_path = f'{save_folder}/filled_similarity_matrix.csv' # saves the similarity matrix containing each query X candidate score

import os
import codecs
import re
import string
import time
import random
import tqdm
import pandas as pd
# import csv 
# from csv import reader
import pickle as pkl

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

class BM25(object):
    def __init__(self, b=0.7, k1=1.6, n_gram:int = 1):
        self.n_gram = n_gram
        self.vectorizer = TfidfVectorizer(max_df=.65, min_df=1,
                                  use_idf=True, 
                                  ngram_range=(n_gram, n_gram))
        
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        start_time = time.perf_counter()
        print(f"Fitting tf_idf vectorizer")
        self.vectorizer.fit(X)
        print(f"Finished tf_idf vectorizer, time : {time.perf_counter() - start_time:0.3f} sec")
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1


my_suffixes = (".txt")
citation_file_paths = []
for r, d, f in os.walk(path_prior_cases):
    for file in f:
        if file.endswith(my_suffixes):
            citation_file_paths.append(os.path.join(r, file))

name_dict = {}
corpus =[]
citation_names = []
for file in sorted(citation_file_paths):
    f = codecs.open(file, "r", "utf-8", errors='ignore')
    text = f.read()
    corpus.append(text)
    citation_names.append(os.path.basename(file))
    name_dict[text] = os.path.basename(file)

my_suffixes = (".txt")
query_file_paths = []
for r, d, f in os.walk(path_current_cases):
    for file in f:
        if file.endswith(my_suffixes):
            query_file_paths.append(os.path.join(r, file))

query_corpus = []
query_names = [] 

#iterate throught the query database list in sorted manner
for file in tqdm.tqdm(sorted(query_file_paths),desc = "query documents"):
    open_file = open(file, 'r', encoding="utf-8")
    text = open_file.read()
    
    raw_str_list = text.splitlines()
    str_list_3 = raw_str_list
    query_corpus.append(''.join(str_list_3))
    query_names.append(os.path.basename(file).zfill(14))
    open_file.close()


with open(true_labels_json, 'r') as f:
    true_labels = json.load(f)  # get the gold labels file
    
bm25 = BM25(n_gram = n_gram)
bm25.fit(corpus)

bm_25_results_dict = {}

# compute BM25 score between each query X candidate document
for i in tqdm.tqdm(range(len(query_corpus))):
    qu = query_corpus[i]
    qu_n = query_names[i]
    
    doc_scores = bm25.transform(qu, corpus)

    assert(int(re.findall(r'\d+',qu_n)[0]) not in bm_25_results_dict)

    bm_25_results_dict[int(re.findall(r'\d+',qu_n)[0])] = {int(re.findall(r'\d+',citation_names[i])[0]) : doc_scores[i] for i in range(len(doc_scores))}

with open(bm25_results_save_dict_path, 'wb') as f:
    pkl.dump(bm_25_results_dict, f)

def obtain_sim_df_from_labels(labels):
    query_numbers = [int(re.findall(r'\d+', i["id"])[0]) for i in labels["Query Set"]]
    relevant_cases = [i["relevant candidates"] for i in labels["Query Set"]]
    relevant_cases = [[int(re.findall(r'\d+', j)[0]) for j in i] for i in relevant_cases] 
    relevant_cases = {i:j for i,j in zip(query_numbers, relevant_cases)}

    candidate_numbers = [int(re.findall(r'\d+', i["id"])[0]) for i in labels["Candidate Set"]]
    candidate_numbers.sort()

    row_wise_dataframe = {}
    for query_number in sorted(list(relevant_cases.keys())):
        relevance_dict = {} # contains 0 for not relevant, 1 for relevant, -1 for self-relevance/citation
        for candidate in candidate_numbers:
            if candidate == query_number:
                relevance_dict[candidate] = -1
            elif candidate in relevant_cases[query_number]:
                relevance_dict[candidate] = 1
            else :
                relevance_dict[candidate] = 0

        row_wise_dataframe[query_number] = relevance_dict

    df = pd.DataFrame(row_wise_dataframe)
    df = df.T
    df.insert(loc=0, column='query_case_id', value=row_wise_dataframe.keys())
    df = df.reset_index(drop=True)
    return df

gold_labels_df = obtain_sim_df_from_labels(true_labels)
sim_df = obtain_sim_df_from_labels(true_labels) # use the gold label file as a backbone to fill the similarity values.
del_columns = [i for i in sim_df.columns if str(i).startswith('Unnamed')]
sim_df = sim_df.drop(columns=del_columns, axis = 1)
column_candidates = list(sim_df.columns)[1:]
column_name = 'query_case_id' if 'query_case_id' in sim_df.columns else 'Unnamed: 0'
for i, query in tqdm.tqdm(enumerate(list(sim_df[column_name].values))):
    assert(sim_df.iloc[i][column_name] == query)
    temp_bm25_scores = [bm_25_results_dict[query][int(i)] for i in column_candidates]
    sim_df.iloc[i] = [float(query)] + temp_bm25_scores

sim_df.to_csv(filled_sim_csv_path)

# compute and save score metrics
output_numbers = evaluate_at_K.get_f1_vs_K(gold_labels_df, sim_df)
with open(f'{save_folder}/output.json', 'w') as f:
    json.dump(output_numbers, f, indent = 4)
