'''
Exps are defined by corpus, checkpoint, top512
'''
import os, sys, json, pickle as pkl
import itertools

def find_folder_and_output(model, corpus, top512):
    for i in os.listdir(f'./exp_results/'):
        path = f'./exp_results/{i}'
        if not os.path.isdir(path): # ignore README.md and other files 
            continue
        with open(path + '/config_file.json', 'r') as f:
            config = json.load(f)
        
        if config["checkpoint"] == model and f'data/corpus/{corpus}' in config["path_prior_cases"] and config["top512"] == top512 :
            with open(path + f'/output.json', 'r') as f:
                output = json.load(f)
            return output
    print(f"exp result for {model, corpus, top512} not found")
    raise RuntimeError

def get_train_exp(model, corpus, top512):
    '''
    Fetches the counterpart train experiment to determine the best (k)
    '''
    train_corpus = corpus.replace('test', 'train')
    for i in EXP_RESULTS:
        if i["model"] == model and i["corpus"] == train_corpus and i["top512"] == top512:
            return i
    print(f'Can\'t find counter-part train experiment for config {model, corpus, top512}')
    raise RuntimeError

def resolve_exp(model, corpus, top512):
    '''
    Fetches results corresponding to the experiment (model, corpus, top512) and appends in the global list EXP_RESULTS
    Example usage : resolve_exp("bert-case-uncased", )
    '''
    output = find_folder_and_output(model, corpus, top512)
    if 'train' in corpus:
        best_k = output['f1_vs_K'].index(max(output['f1_vs_K']))
        return_dict = {
                'model' : model,
                'corpus' : corpus,
                'top512' : top512,
                'best_k' : best_k+1,
                'recall_vs_K' : output['recall_vs_K'][best_k],
                'precision_vs_K' : output['precision_vs_K'][best_k],
                'f1_v_k' : output['f1_vs_K'][best_k],
            }
        return return_dict

    elif 'test' in corpus:
        train_exp = get_train_exp(model, corpus, top512)
        best_k = train_exp['best_k']
        return_dict = {
                'model' : model,
                'corpus' : corpus,
                'top512' : top512,
                'best_k_train' : best_k,
                'recall_vs_K' : output['recall_vs_K'][best_k-1],    # as best_k is saved starting from index 1
                'precision_vs_K' : output['precision_vs_K'][best_k-1],
                'f1_v_k' : output['f1_vs_K'][best_k-1],
            }
        return return_dict
    
    else :
        print('corpus name \'{corpus}\' does not contain either test or train')
    raise RuntimeError

def show(i):
    i['precision_vs_K'] = f"{round(i['precision_vs_K']*100, 2)}%"
    i['recall_vs_K'] = f"{round(i['recall_vs_K']*100, 2)}%"

    i['f1_v_k'] = round(i['f1_v_k']*100, 2)
    i['f1_v_k'] = f"{i['f1_v_k']}%"

    if SHOW_MODE == 'TRAIN':
        if 'best_k' in i:
            print(i)
    if SHOW_MODE == 'TEST':
        if 'best_k_train' in i:
            print(i)

if __name__ == '__main__':
    model_names = ['bert-base-uncased', './model_checkpoints/bert_finetuned/', 'distilbert-base-uncased', './model_checkpoints/distilbert_finetuned/', 'law-ai/InCaseLawBERT', 'law-ai/InLegalBERT']
    # model_names = ['bert-base-uncased', 'distilbert-base-uncased', 'law-ai/InLegalBERT', 'law-ai/InCaseLawBERT']
    # corpus_names = ['ik_train', 'ik_test', 'sentence_removed/ik_train', 'sentence_removed/ik_test'] # for ILPCR dataset
    corpus_names = ['ik_train', 'ik_test'] # for ILPCR dataset
    # corpus_names = ['sentence_removed/ik_train', 'sentence_removed/ik_test']
    # corpus_names = ['COLIEE2021/train', 'COLIEE2021/test']    # for coliee results
    # top512_values = ["True", "False"]
    top512_values = ["True"]

    EXP_RESULTS = []
    for model, corpus, top512 in itertools.product(*[model_names, corpus_names, top512_values]):
        EXP_RESULTS.append(resolve_exp(model, corpus, top512))
    
    # SHOW_MODE = 'TRAIN'
    SHOW_MODE = 'TEST'
    for i in EXP_RESULTS:
        show(i)
