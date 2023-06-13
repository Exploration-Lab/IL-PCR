# %%
import re, sys, os
import numpy as np, pandas as pd
import json
import pickle as pkl
#import matplotlib.pyplot as plt
from tqdm import tqdm
#from tqdm.notebook import tqdm_notebook

from sklearn.metrics import (accuracy_score, f1_score, classification_report)
from transformers import BertTokenizer, BertModel
from sentence_transformers import models
import torch, gc

# %%
torch.cuda.empty_cache()
gc.collect()

# %%
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"running on device {device}")

tokenizer = []

def get_query_candidate_cases():
    # get query and candidate case numbers
    query_nos = []
    candidate_nos = []

    # fetch case numbers
    for item in os.listdir(test_dir_path + f'/query'):
        temp = int(item.strip('.txt'))
        query_nos.append(temp)

    for item in os.listdir(test_dir_path + f'/candidate'):
        temp = int(item.strip('.txt'))
        candidate_nos.append(temp)

    query_nos.sort()
    candidate_nos.sort()
    return query_nos, candidate_nos

# %%

# %%
# load segments
# with open(test_dir_path + f'/segment_dictionary_ik_test.sav', 'rb') as f:
#     out = pkl.load(f, encoding="latin-1")
#     query_docs_segmented = out['dict_query']
#     candidate_docs_segmented = out['dict_candidate']
#     f.close()

# %%
embedding_chunk_size = 64
def get_text_embedding(text, model):
    model.eval()
    with torch.no_grad():
        hidden_states = []
        for i in range(0,len(text), embedding_chunk_size):
            text_chunk = text[i : min(i+embedding_chunk_size, len(text))]
            hidden_state = model.encode(text_chunk)
            hidden_states.append(torch.from_numpy(hidden_state))
            del text_chunk, hidden_state  # cleaning ??, not the cause for memory overflows, check after fixing batch size issue 
    ret = torch.vstack(hidden_states).squeeze(dim=1)
    del hidden_states
    return ret

def get_embeddings_dict(dic:dict,model):
    keys = list(dic.keys())
    end_array, all_chunks = [], []
    __len__ = 0
    for i in dic.values():
        end_array.append(__len__)
        if len(i) == 0:
          all_chunks.extend([" "])
          __len__ += 1
        else:
          all_chunks.extend(i)
          __len__ += len(i)
    end_array.append(__len__)
    
    all_embeddings = []
    for chunk in tqdm(range(0, len(all_chunks), embedding_chunk_size), desc='Text chunks'):
        embedding_ = get_text_embedding(all_chunks[chunk : chunk + embedding_chunk_size], model)
        all_embeddings.extend(list(embedding_))
    
    output = {}
    # for i, key in enumerate(keys):
    #     output[key] = torch.stack(all_embeddings[end_array[i] : end_array[i+1]])
    for i, key in enumerate(keys):
        try : 
          output[key] = torch.stack(all_embeddings[end_array[i] : end_array[i+1]])
        except:
            print(f"error at iter {i}\nerror for embeddings {all_embeddings[end_array[i] : end_array[i+1]]}\n end_array[i] {end_array[i]}, end_array[i+1] {end_array[i+1]}, ")
            
    return output


def get_query_candidate_embeddings(model,segment_dictionary_data):
    # with open(test_dir_path +'/'+segment_dictionary_name, 'rb') as f:
    #     out = pkl.load(f, encoding="latin-1")
    query_docs_segmented = segment_dictionary_data['dict_query']
    candidate_docs_segmented = segment_dictionary_data['dict_candidate']
    print("Starting for query_docs_segmented")
    q = get_embeddings_dict(query_docs_segmented,model)
    print("Starting for candidate_docs_segmented")
    c = get_embeddings_dict(candidate_docs_segmented,model)

    return q, c

def relevance(args):
    query_num, candidate_num,SIMILARITY_MATRICES,query_embeddings,candidate_embeddings = args

    #find pairwise similarity
    query_embed = query_embeddings[query_num]
    candidate_embed = candidate_embeddings[candidate_num]
    # print("For query:",query_num," , Query_embed size:",len(query_embed))
    # print("For candidate:",candidate_num," , candidate_embed size:",len(candidate_embed))

    q = query_embed.cuda()
    c = candidate_embed.cuda()
    q = q / q.norm(dim = 1)[:, None ]
    c = c / c.norm(dim = 1)[:, None ]

    similarity_matrix = torch.matmul(q, c.T)
    # SIMILARITY_MATRICES[int(query_num)][int(candidate_num)] = similarity_matrix
    return torch.max(similarity_matrix).item()

def run_relevance(SIMILARITY_MATRICES,query_embeddings,candidate_embeddings):
    score_dict = {}
    for query_num in tqdm(query_embeddings.keys(), desc = 'Get similarity matrices'):
        SIMILARITY_MATRICES[int(query_num)] = {}

        temp_scores = []
        for candidate_num in candidate_embeddings.keys():
            __score__ = relevance((query_num, candidate_num,SIMILARITY_MATRICES,query_embeddings,candidate_embeddings))
            temp_scores.append(__score__)
            
        candidate_list = list(candidate_embeddings.keys())
        score_dict[query_num] = {candidate_list[count]:i for count, i in enumerate(temp_scores)}

    return score_dict


def similarity(model,i,model_id,segment_dictionary_name,segment_dictionary_data):
    model = model.to(device)
    print("Starting Similarity-------> ",i,",",segment_dictionary_name,",",model_id)
    SIMILARITY_MATRICES = {}
    query_embeddings, candidate_embeddings = get_query_candidate_embeddings(model,segment_dictionary_data)
    scores_dict = run_relevance(SIMILARITY_MATRICES,query_embeddings,candidate_embeddings)
    print("Similarity Calculation Finished------------>",i,",",segment_dictionary_name,",",model_id)
    return scores_dict


with open('./SBERT_input_details.json','r') as f:
    input_details = json.load(f)
test_dir_path = input_details['data_dir_path'] #test_dir_path = r"./indiankanoon/train"
query_segment_dictionary_path = input_details['query_segment_dictionary_path']
candidate_segment_dictionary_path = input_details['candidate_segment_dictionary_path']
model_type = input_details['model_type']  #"bert"
model_id = input_details['model_id']   #"0"
dataset =input_details['dataset'] #"ik"
split_type = input_details['split_type']  #"train"
sim_csv_root = "./Sim_CSVs/"+dataset+"/"+model_type+"_base"+"/"
os.makedirs(sim_csv_root,exist_ok=True)
csv_file_name = dataset+"_"+split_type+"_"+model_type+"_"+str(model_id)+".csv"
if model_id == 0:
    # model_name = "bert-base-uncased" or "distilroberta-base"
    model_name = input_details['model_name']
    max_seq_length = 32
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model]) # model = SentenceTransformer(model_path, device='cuda')
else:
    model_path = input_details['model_path']
    model = SentenceTransformer(model_path, device='cuda')
    
model = model.to(device)

#load segment_dictionary_data
with open(query_segment_dictionary_path, 'rb') as f:
    query_segment_dictionary_data = pkl.load(f, encoding="latin-1")
with open(candidate_segment_dictionary_path, 'rb') as f:
    candidate_segment_dictionary_data = pkl.load(f, encoding="latin-1")

query_data = query_segment_dictionary_data['query_data']
candidate_data = candidate_segment_dictionary_data['candidate_data']
segment_dictionary_data = {"dict_query":query_data,"dict_candidate":candidate_data}
segment_dictionary_name =  str(query_segment_dictionary_path.split("/")[-1])+"_and_" + str(candidate_segment_dictionary_path.split("/")[-1])


print("Number of query docs: ",len(query_data))
print("Number of candidate docs: ",len(candidate_data))
scores_dict = similarity(model,0,model_id,segment_dictionary_name,segment_dictionary_data)

print("Saving Similarity CSV--------->",segment_dictionary_name,",",model_id)

all_candidates = sorted(list(scores_dict[list(scores_dict.keys())[0]].keys()))
print("Length of all_candidates: ",len(all_candidates))
test_SBERT_sim = dict()
for query, sbert_score_dict in scores_dict.items():
    test_SBERT_sim[query] = list()
    for candidate in all_candidates:
        test_SBERT_sim[query].append(sbert_score_dict[candidate])

print("sim_csv will be saved at ",sim_csv_root)
pd.DataFrame.from_dict(data=test_SBERT_sim, orient='index').to_csv(sim_csv_root+csv_file_name,header=all_candidates)
df = pd.read_csv(sim_csv_root+csv_file_name)
df.columns = ['query_case_id'] + list(df.columns[1:])
df.to_csv(sim_csv_root+csv_file_name,index=False)
print("Similarity CSV saved at: ",sim_csv_root+csv_file_name)
print("***************Finished Similarity for model: ",segment_dictionary_name,",",model_id," ******************")
print("Finished")