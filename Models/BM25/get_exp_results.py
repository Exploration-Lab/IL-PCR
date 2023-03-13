'''
Print the results of all the experiments in the `exp_results/` folder.
Fetches the best (k) from the train experiment and reports the testset F1 on the appropriate (k).
'''

import os, sys, glob
import json

def get_train_counterpart(exp_name):
    assert(('test' in exp_name) and ('train' not in exp_name))  # for the test experiments
    return exp_name.replace("test", "train")

def get_exp_result(exp_name):
    '''
    Fetches the experiment folder matching the experiment_name and appends it to the global (EXP_RESULTS) list.
    Example usage : get_exp_result("ik_train_atomic"), get_exp_result("ik_test"), 
    '''
    path = f'./data/corpus/{exp_name}/'
    output_files = {}   # contains all the folders (at all n_grams) of the queried experiment
    for config_file_path in glob.glob('./exp_results/*/config_file.json'):
        with open(config_file_path, 'r') as f:
            config = json.load(f)
            if config['path_prior_cases'].startswith(path):
                output_dict_path = config_file_path[:config_file_path.rfind('/')] + f'/output.json'

                assert os.path.isfile(output_dict_path), f"exp name \"{exp_name}\" exists but does not contain an output file. Experiment is either incomplete or corrupt."

                with open(output_dict_path, 'r') as f2:
                    output_dict = json.load(f2)
                output_files[config['n_gram']] = output_dict    # contains results corresponding to each (n_gram)

    if 'train' in exp_name:
        for n_gram, res in output_files.items():
            best_k = res['f1_vs_K'].index(max(res['f1_vs_K']))
            return_dict = {
                'exp_name' : exp_name,
                'n_gram' : n_gram,
                'best_k' : best_k+1,
                'precision_vs_K' : res['precision_vs_K'][best_k],
                'recall_vs_K' : res['recall_vs_K'][best_k],
                'f1_v_k' : res['f1_vs_K'][best_k],
            }
            EXP_RESULTS.append(return_dict)
    
    
    elif 'test' in exp_name:
        # the (best_k) value for test experiments requires the (best_k) value obtained from the train counter-part of the same experiment
        counter_trainname = get_train_counterpart(exp_name)
        for n_gram, res in output_files.items():
            train_entry = [i for i in EXP_RESULTS if ((i['exp_name'] == counter_trainname) and (i['n_gram'] == n_gram))]
            if(len(train_entry) == 0):
                print(f'Train counterpart experiment for {exp_name} : {counter_trainname}, n_gram : {n_gram} not found!\nRerunning the train counterpart experiment should do the trick.')
                raise RuntimeError
            assert len(train_entry) == 1, f"experiment {exp_name} has mulitple counterparts : {train_entry}. Please delete redundant folders."   # only 1 counterpart experiment
            best_k_train = train_entry[0]['best_k']

            return_dict = {
                'exp_name' : exp_name,
                'n_gram' : n_gram,
                'best_k_train' : best_k_train,
                'precision_vs_K' : res['precision_vs_K'][best_k_train-1],
                'recall_vs_K' : res['recall_vs_K'][best_k_train-1], # as the (best_K) value is saved starting from index = 1
                'f1_v_k' : res['f1_vs_K'][best_k_train-1],
            }
            EXP_RESULTS.append(return_dict)
    
    return

def show(i):
    i['precision_vs_K'] = f"{round(i['precision_vs_K']*100, 2)}%"
    i['recall_vs_K'] = f"{round(i['recall_vs_K']*100, 2)}%"

    i['f1_v_k'] = round(i['f1_v_k']*100, 2)
    i['f1_v_k'] = f"{i['f1_v_k']}%"

    if SHOW_MODE == 'TRAIN' and 'best_k' in i:
        print(i)
        
    if SHOW_MODE == 'TEST' and 'best_k_train' in i:
        print(i)
    
if __name__ == '__main__':

    # list of experiments whose results are to be fetched 
    exp_ILPCR = [
        'ik_train', 'ik_test', 
      'ik_train_atomic', 'ik_test_atomic', 
    'ik_train_events', 'ik_test_events', 
    'ik_train_iouf', 'ik_test_iouf',
    'ik_train_RR', 'ik_test_RR',
     ]      

    # for the citation-sentence removed ILPCR
    exp_ILPCR_sentence_removed = [f"sentence_removed/" + i for i in exp_ILPCR]  
    
    exp_coliee = [
        'COLIEE2021/train', 
        'COLIEE2021/test', 
        "COLIEE2021_train_events", 
        "COLIEE2021_test_events", 
        "COLIEE2021_train_atomic", 
        "COLIEE2021_test_atomic",
        "COLIEE2021_train_iou_filtered",
        "COLIEE2021_test_iou_filtered",
        "COLIEE2021_RR_train", 
        "COLIEE2021_RR_test",
    ]

    EXP_RESULTS = []

    # for i in exp_ILPCR + exp_ILPCR_sentence_removed + exp_coliee:
    for i in exp_ILPCR :
        _ = get_exp_result(i)
    
    # SHOW_MODE = 'TRAIN'   # print the train/test set results in particular
    SHOW_MODE = 'TEST'  

    for i in EXP_RESULTS:
        show(i)
