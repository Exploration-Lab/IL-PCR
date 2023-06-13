# %%
import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

# %%
# load IOU_filtered_input_details.json
with open(r"./IOU_filtered_input_details.json", 'rb') as f:
    input_details = json.load(f)
# load segment dictionary
dataset = input_details['dataset']  # indiankanoon,COLIEE etc
split_type = input_details['split_type']  # train,dev,test

# load segment dictionary
seg_data_dir_path = input_details['seg_data_dir_path']
# load segment dictionary for cadidate
seg_path_candi = seg_data_dir_path+"/"+dataset+"/"+split_type + \
    "/segment_dictionary_"+split_type+"_"+dataset+"_candidate.sav"
with open(seg_path_candi, 'rb') as f:
    candi_seg_dict = pkl.load(f)
# load segment dictionary for query
seg_path_query = seg_data_dir_path+"/"+dataset+"/"+split_type + \
    "/segment_dictionary_"+split_type+"_"+dataset+"_query.sav"
with open(seg_path_query, 'rb') as f:
    query_seg_dict = pkl.load(f)

# load event doc line text
event_doc_line_text_dir_path = input_details['event_doc_line_text_dir_path']
# load event doc line text for cadidate
event_doc_line_text_path_candi = event_doc_line_text_dir_path+"/"+dataset + \
    "/"+split_type + "/event_doc_line_text_"+split_type+"_"+dataset+"_candidate.pkl"
with open(event_doc_line_text_path_candi, 'rb') as f:
    candidate_event_doc_txt = pkl.load(f)
# load event doc line text for query
event_doc_line_text_path_query = event_doc_line_text_dir_path+"/"+dataset + \
    "/"+split_type + "/event_doc_line_text_"+split_type+"_"+dataset+"_query.pkl"
with open(event_doc_line_text_path_query, 'rb') as f:
    query_event_doc_txt = pkl.load(f)

print("Loaded seg_path_query:", seg_path_query)
print("Loaded seg_path_candi:", seg_path_candi)
print("Loaded event_doc_line_text_path_query:", event_doc_line_text_path_query)
print("Loaded event_doc_line_text_path_candi:", event_doc_line_text_path_candi)

# %%
# sorted list of all queries and candidates
all_queries = sorted(list(query_seg_dict['dict_query'].keys()))
all_candidates = sorted(list(candi_seg_dict['dict_candidate'].keys()))
print(len(all_queries))
print(len(all_candidates))

# %%
# making matrix of all common events between queries and candidates
qc_mat = list()
for q in tqdm(all_queries,desc="Matrix Calculation:"):
    # print(q)
    qc_events = list()
    # q_events = {tuple(e) for e in query_seg_dict['dict_query'][q]}
    q_events = set(query_seg_dict['dict_query'][q])
    for c in all_candidates:
        # print(c)
        if q != c:
            # c_events = {tuple(e) for e in candi_seg_dict['dict_candidate'][c]}
            c_events = set(candi_seg_dict['dict_candidate'][c])
            qc_events.append(q_events.intersection(c_events))
        else:
            # print("same",q,c)
            qc_events.append(set())
        # break
    qc_mat.append(qc_events)

# %%
# check length of qc_mat = number of queries
print(len(qc_mat))
# check length of all lists qc_mat = number of candidates
ct = set()
for ql in qc_mat:
    ct.add(len(ql))
print(ct)

# %%
# unique events in a query
query_events = {}
for i in range(len(all_queries)):
    q_id = all_queries[i]
    # print(q_id)
    qc_events = qc_mat[i]
    qc_common = set().union(*qc_events)
    query_events[q_id] = qc_common
    # break
# check length of query_events = number of queries
print(len(query_events))

# %%
# unique events in a candidate
candidate_events = {}
for i in range(len(all_candidates)):
    c_id = all_candidates[i]
    # print(c_id)
    cq_events = list()
    for row in qc_mat:
        cq_events.append(row[i])
    cq_common = set().union(*cq_events)
    candidate_events[c_id] = cq_common
    # break

# check length of candidate_events = number of candidates
print(len(candidate_events))

# %%
# convert common query events into the source sentences
query_text = {}
for q, eves in query_events.items():
    q_txt = {}
    for eve in list(eves):
        if eve in query_event_doc_txt:
            if q in query_event_doc_txt[eve]:
                q_txt.update(query_event_doc_txt[eve][q])
    q_lst = []
    for key in sorted(list(q_txt.keys())):
        # print(key,"::",q_txt[key])
        q_lst.append(q_txt[key])
    query_text[q] = q_lst
    # break

# %%
# convert common candidate events into the source sentences
candidate_text = {}
for c, eves in candidate_events.items():
    c_txt = {}
    for eve in list(eves):
        if eve in candidate_event_doc_txt:
            if c in candidate_event_doc_txt[eve]:
                c_txt.update(candidate_event_doc_txt[eve][c])
    c_lst = []
    for key in sorted(list(c_txt.keys())):
        # print(key,"::",c_txt[key])
        c_lst.append(c_txt[key])
    candidate_text[c] = c_lst

# %%
# save all text in dictionary
event_text_dict = dict()
event_text_dict['dict_query'] = query_text
event_text_dict['dict_candidate'] = candidate_text


# %%
# save file to location
output_path = input_details['output_dir']+"/"+dataset+"/"+split_type
output_file_path = output_path+"/IOU_filtered_text_dict_"+dataset+"_"+split_type+".sav"
os.makedirs(output_path, exist_ok=True)
with open(output_file_path, 'wb') as f:
    try:
        pkl.dump(event_text_dict, f)
    except:
        print("Couldn't save")
    f.close()
print("Saved IOU_filtered_text_dict_"+dataset +
      "_"+split_type+".sav at:", output_file_path)
