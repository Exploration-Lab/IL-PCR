import re, sys, os, json, time
import numpy as np, pandas as pd, pickle as pkl
from tqdm import tqdm
import evaluate_at_K
import spacy

import torch, gc    # clean torch cuda cache before starting
torch.cuda.empty_cache()
gc.collect()

# make save folder 
assert(len(sys.argv) > 1)
current_time = '_'.join(time.ctime().split()) + '_' + str(os.getpid())    # process pid is appended to ensure that experiments spawned at the same time don't overwrite each other.
save_folder = f'./exp_results/EXP_{current_time}'
os.makedirs(save_folder)

from transformers import AutoTokenizer, AutoModel
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"running on device {device}")

with open(sys.argv[1], 'r') as f:
    config_dict = json.load(f)
with open(save_folder + f'/config_file.json', 'w') as f:
    json.dump(config_dict, f, indent = 4)

path_candidate_cases = config_dict['path_prior_cases']
path_query_cases = config_dict['path_current_cases']
true_labels_json = config_dict['true_labels_json']
checkpoint = config_dict['checkpoint']
top512 = config_dict['top512']
dir_path = path_candidate_cases[:path_candidate_cases.rfind('/')]

assert(os.path.isdir(path_candidate_cases))
assert(os.path.isdir(path_query_cases))
assert(os.path.isfile(true_labels_json))
assert(top512 == "True" or top512 == "False")

scores_save_path = f'{save_folder}/scores.json'     # dictionary containing the score between each query X candidate pair
filled_sim_csv_path = f'{save_folder}/filled_similarity_matrix.csv' # csv containing the similarity matrix

# get all query and candidate cases
query_cases_nos = os.listdir(dir_path + '/query')
query_docs = {}
for i in query_cases_nos:
    temp_path = dir_path + '/query/' + i
    with open(temp_path, 'r', encoding = "utf-8") as f:
        content = f.read()
    query_numbers = re.findall(r'\d+', i)
    if len(query_numbers) == 0: # ignore .gitkeep files
        continue
    query_docs[int(query_numbers[0])] = content

candidate_cases_nos = os.listdir(dir_path + '/candidate')
candidate_docs = {}
for i in candidate_cases_nos:
    temp_path = dir_path + '/candidate/' + i
    with open(temp_path, 'r', encoding = 'utf-8') as f:
        content = f.read()
    candidate_numbers = re.findall(r'\d+', i)
    if len(candidate_numbers) == 0: # ignore .gitkeep files
        continue
    candidate_docs[int(candidate_numbers[0])] = content

if 'coliee' in dir_path.lower():
    padding_len = 6 
elif 'ik' in dir_path.lower():
    padding_len = 10
else :
    print("You appear to be using a new dataset, so set the padding_len explicitly. The padding length is the length of the numerical portion of the case id. The COLIEE dataset has a padding length of 6 and the ILPCR dataset has a padding length of 10.")
    raise RuntimeError

def get_query_candidate_cases():
    query_nos = []
    candidate_nos = []

    # fetch case numbers
    for item in os.listdir(dir_path + f'/query'):
        if not item.endswith('.txt'):
            continue
        temp = int(item.strip('.txt'))
        query_nos.append(temp)

    for item in os.listdir(dir_path + f'/candidate'):
        if not item.endswith('.txt'):
            continue
        temp = int(item.strip('.txt'))
        candidate_nos.append(temp)

    query_nos.sort()
    candidate_nos.sort()
    return query_nos, candidate_nos

# load model checkpoint
print(f"Loading model {checkpoint}")
checkpoint_name = checkpoint[checkpoint.rfind('/')+1:]
if 'distilbert' in checkpoint:
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
elif 'bert' in checkpoint:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
elif 'longformer' in checkpoint:
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
else :
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if checkpoint == 'law-ai/InCaseLawBERT':
        tokenizer.model_max_length = 510    # IncaseLawBERT checkpoint on hugging face doesn't have model_max_length setup.

model = AutoModel.from_pretrained(checkpoint)
model = model.to(device)

# get case segments
def get_base_case(k: int):
    """
    Read a base case.
    Example Usage : get_base_case(28)

    Args:
      k: Base case number
    Returns:
      One segmented base case.
    """
    if(padding_len == 10):
      path_to_file = dir_path + f"/query/{k:010d}.txt" # in query folder
    elif(padding_len == 6):
      path_to_file = dir_path + f"/query/{k:06d}.txt" # in query folder
    
    with open(path_to_file, 'r', encoding = 'utf-8') as f:
        contents = f.read()

    contents = contents.replace('\n',' ')
    base_case = ' '.join(contents.split())
    
    return base_case

def get_candidate_case( base_case_num: int, k: int):
    if(padding_len == 10):
      path_to_file = dir_path + f"/candidate/{k:010d}.txt"  # in candidate folder
    elif(padding_len == 6):
      path_to_file = dir_path + f"/candidate/{k:06d}.txt"  # in candidate folder
  
    with open(path_to_file, 'r', encoding = 'utf-8') as f:
        contents = f.read()

    contents = contents.replace('\n',' ')
    candidate_case = ' '.join(contents.split())
    
    return candidate_case

def example2seg(docs, max_sentences_per_segment : int = 1 , stride : int = 1, top_N_segments : int = int(1e6)):
    """
    Taken from https://github.com/neuralmind-ai/coliee
    Receives a document and segment it.
    Example Usage : example2seg(get_candidate_case(query_nos[0], candidate_nos[0]), 4, 2, 3)

    Args:
      docs: Document
      max_sentences_per_segment: number of sentences in each segment
      stride: stride
      top_N_segments : return only (top_N_segments) number of segments.
    Returns:
      One segmented document.
    """

    doc = nlp(docs)
    sentences = [sent.text.strip() for sent in doc.sents]
    segments = []

    for i in range(0, len(sentences), stride):
      segment = ' '.join(sentences[i:i + max_sentences_per_segment])
      segments.append(segment)
      if i + max_sentences_per_segment >= len(sentences) or len(segments) >= top_N_segments :
          break

    return segments

def save_top_N_segments(query_nos, candidate_nos, top_N = 25):
    # if os.path.isfile(dir_path + f'/segment_dictionary.sav'):
    #     with open(dir_path + f'/segment_dictionary.sav', 'rb') as f:
    #         ans = pkl.load(f)
    #     return ans

    dict_query = {}
    dict_candidate = {}
    
    for i in tqdm(range(len(query_nos)), desc = 'Saving query segments'):
        base_case = query_nos[i]
        base_doc = get_base_case(base_case)
        base_segments = example2seg(base_doc, 10, 5, top_N_segments = top_N)
        dict_query[base_case] = base_segments

    for i in tqdm(range(len(candidate_nos)), desc = 'Saving candidate segments'):
        candidate_case = candidate_nos[i]
        candidate_doc = get_candidate_case(query_nos[0], candidate_case)
        candidate_segments = example2seg(candidate_doc, 10, 5, top_N_segments = top_N)
        dict_candidate[candidate_case] = candidate_segments
    
    ans = {'dict_query' : dict_query, 'dict_candidate' : dict_candidate}

    # with open(dir_path + f'/segment_dictionary.sav', 'wb') as f:
    #     try : 
    #         pkl.dump(ans, f)
    #     except:
    #         print("Couldn't save")

    return ans

# make segments
global nlp
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
nlp.max_length = 3000000

query_nos, candidate_nos =  get_query_candidate_cases()
_ = save_top_N_segments(query_nos, candidate_nos, np.inf)

################## get embeddings ##################
embedding_chunk_size = 64
def get_text_embedding(text, tokenizer, model):
    model.eval()
    with torch.no_grad():
        hidden_states = []
        for i in range(0,len(text), embedding_chunk_size):
            text_chunk = text[i : min(i+embedding_chunk_size, len(text))]
            inputs = tokenizer(text_chunk, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1][:,0,:].to('cpu')
            hidden_states.append(hidden_state)

            del text_chunk, inputs, outputs, hidden_state

    ret = torch.vstack(hidden_states).squeeze(dim=1)
    del hidden_states
    return ret

def get_embeddings_dict(dic:dict):
    '''
    Computes the dense vector representation of the cases present in (dic) based on the globally set (top512) variable.
    If (top512) is set to True, the document is passed as it is to the transformer model and longer documents are truncated.
    If (top512) is set to False, the document is divided into segments and passed into the transformer model. The final representation is obtained by stacking the segment-wise vectors.
    '''
    keys = list(dic.keys())
    output = {}
    if top512 == "False":
      end_array, all_chunks = [], []
      __len__ = 0
      for key, chunk in dic.items():
        end_array.append(__len__)
        
        if len(chunk) == 0:
          all_chunks.extend([" "])
          __len__ += 1
        else:
          all_chunks.extend(chunk)
          __len__ += len(chunk)
      end_array.append(__len__)

      for i in range(len(end_array)-1):
        assert(end_array[i] != end_array[i+1])  # so that empty tensor doesn't go to torch.stack()
      
      all_embeddings = []
      for chunk in tqdm(range(0, len(all_chunks), embedding_chunk_size), desc='Text chunks'):
          embedding_ = get_text_embedding(all_chunks[chunk : chunk + embedding_chunk_size], tokenizer, model)
          all_embeddings.extend(list(embedding_))
      
      output = {}
      for i, key in enumerate(keys):
          output[key] = torch.stack(all_embeddings[end_array[i] : end_array[i+1]])

    elif top512 == "True":
      for q_number, text in tqdm(dic.items()):
        hidden_state = get_text_embedding([text], tokenizer, model)
        output[q_number] = hidden_state
    
    else :
      print(f"Not expected : top512 is {top512}")
      raise RuntimeError

    return output

def get_query_candidate_embeddings():
    if top512 == "False":
        with open(dir_path + f'/segment_dictionary.sav', 'rb') as f:    # load the segmented dictionaries
          out = pkl.load(f)
        query_docs_segmented = out['dict_query']
        candidate_docs_segmented = out['dict_candidate']
      
        # Obtain vector representations using the segment dictionary.
        q = get_embeddings_dict(query_docs_segmented)
        c = get_embeddings_dict(candidate_docs_segmented)

        q = { i:v.numpy()  for i,v in q.items()}
        c = { i:v.numpy()  for i,v in c.items()}

    if top512 == "True":
        # Obtain vector representations using the case text.
        q = get_embeddings_dict(query_docs)
        c = get_embeddings_dict(candidate_docs)
        q = { i:v.numpy()  for i,v in q.items()}
        c = { i:v.numpy()  for i,v in c.items()}

    return q, c
  
query_embeddings, candidate_embeddings = get_query_candidate_embeddings()
del model, tokenizer  ### clear up the GPU

################## compute relevance ##################
def relevance(args):
    query_num, candidate_num = args

    #find pairwise similarity
    with torch.no_grad():
        query_embed = query_embeddings[query_num]
        candidate_embed = candidate_embeddings[candidate_num]

        q = torch.Tensor(query_embed).to(device)    # shift matrix multiplication to the GPU
        c = torch.Tensor(candidate_embed).to(device)

        q = q / q.norm(dim = 1)[:, None]
        c = c / c.norm(dim = 1)[:, None]

        similarity_matrix = torch.matmul(q, c.T)
        ret = torch.max(similarity_matrix).cpu().item()
        del q, c, similarity_matrix

    return ret


def run_relevance():
    score_dict = {}
    for query_num in tqdm(query_embeddings.keys(), desc = 'Getting similarity matrices'):
        temp_scores = []
        for candidate_num in candidate_embeddings.keys():
            __score__ = relevance((query_num, candidate_num))
            temp_scores.append(__score__)
        candidate_list = list(candidate_embeddings.keys())
        score_dict[query_num] = {candidate_list[count]:i for count, i in enumerate(temp_scores)}
    
    return score_dict

score_dict = run_relevance()
with open(scores_save_path, 'w') as f:  # save scores
  json.dump(score_dict, f, indent = 4)

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

with open(true_labels_json, 'r') as f:
    true_labels = json.load(f)  # get the gold labels file

gold_labels_df = obtain_sim_df_from_labels(true_labels)
sim_df = obtain_sim_df_from_labels(true_labels) # use the gold label file as a backbone to fill the similarity values.
column_name = 'query_case_id' if 'query_case_id' in sim_df.columns else 'Unnamed: 0'
for i, query in tqdm(enumerate(list(sim_df[column_name].values))):
    assert(sim_df.iloc[i][column_name] == query)
    sim_df.iloc[i] = [float(query)] +list(score_dict[query].values())

sim_df.to_csv(filled_sim_csv_path)

output_numbers = evaluate_at_K.get_f1_vs_K(gold_labels_df, sim_df) # compute evaluation metrics
with open(f'{save_folder}/output.json', 'w') as f:
    json.dump(output_numbers, f, indent = 4)
