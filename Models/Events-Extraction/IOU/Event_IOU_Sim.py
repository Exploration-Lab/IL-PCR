# %%
import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

# load Event_IOU_Sim_input_details.json
with open('Event_IOU_Sim_input_details.json') as f:
    input_details = json.load(f)
dataset = input_details['dataset']
split_type = input_details['split_type']  # test,train
path_sim = "./Sim_CSVs/"+dataset+"/"
os.makedirs(os.path.dirname(path_sim), exist_ok=True)
path_sim_csv = path_sim +"/"+dataset+"_"+split_type+'_IOU_sim.csv'
query_seg = input_details['query_segment_dictionary_path']
candidate_seg = input_details['candidate_segment_dictionary_path']
print("dataset:",dataset)
print("split_type:",split_type)
print("query_seg_path:",query_seg)
print("candidate_seg_path:",candidate_seg)
# %%
# open query_seg and candidate_seg
with open(query_seg, 'rb') as f:
    query_segment_dict = pkl.load(f)
    f.close()
with open(candidate_seg, 'rb') as f:
    candidate_segment_dict = pkl.load(f)
    f.close()
segment_dict = {"dict_query": query_segment_dict['dict_query'],
                "dict_candidate": candidate_segment_dict['dict_candidate']}

# Jaccard Similarity


def Jaccard_Similarity(doc1, doc2):
    # List the unique words in a document
    words_doc1 = set(doc1)
    words_doc2 = set(doc2)

    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)

    # Calculate Jaccard similarity score
    # using length of intersection set divided by length of union set
    if len(intersection) == 0:
        return 0
    return float(len(intersection)) / len(union)


# %%
tot_jaccard_dict = dict()
tot_jaccard_query_case_rank = dict()
for query in tqdm(segment_dict['dict_query'].keys(), desc="Jacc Sim:"):
    jaccard = dict()
    for citation in segment_dict['dict_candidate'].keys():
        # if citation != query:
        doc1 = segment_dict['dict_query'][query]
        doc2 = segment_dict['dict_candidate'][citation]
        res = Jaccard_Similarity(doc1, doc2)
        jaccard[citation] = res
    t = list(jaccard.items())
    jaccard_sorted = sorted(t, key=lambda x: x[1], reverse=True)
    jaccard_case_rank = []
    for tup in jaccard_sorted:
        jaccard_case_rank.append(tup[0])
    tot_jaccard_dict[query] = jaccard
    tot_jaccard_query_case_rank[query] = jaccard_case_rank
    # break

all_queries = sorted(list(segment_dict["dict_query"].keys()))
all_candidates = sorted(list(segment_dict["dict_candidate"].keys()))
print("Length of queries:", len(all_queries))
print("Length of candidates:", len(all_candidates))
# %%
IOU_sim = dict()
for query in tqdm(all_queries):
    jacc_score_dict = tot_jaccard_dict[query]
    # print(query)
    IOU_sim[query] = list()
    for candidate in all_candidates:
        IOU_sim[query].append(jacc_score_dict[candidate])

# %%
pd.DataFrame.from_dict(data=IOU_sim, orient='index').to_csv(
    path_sim_csv, header=all_candidates)
df = pd.read_csv(path_sim_csv)
df.columns = ['query_case_id'] + list(df.columns[1:])
df.to_csv(path_sim_csv, index=False)
