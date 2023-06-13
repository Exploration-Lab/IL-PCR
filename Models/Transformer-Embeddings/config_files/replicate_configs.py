import os, sys, json

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    folder_name = sys.argv[1]
    assert(len(os.listdir(folder_name)) == 1)
    
    with open(f'./{folder_name}/config_1.json', 'r') as f:
        config = json.load(f)
        assert('test' in config["path_prior_cases"])
        assert(config['top512'] == "False")

    for i in range(2,5):
        temp_config = config
        if i%2 == 0:
            temp_config['top512'] = "True"
        else :
            temp_config['top512'] = "False"

        if i >= 3:
            temp_config['path_prior_cases'] = temp_config['path_prior_cases'].replace('test', 'train')
            temp_config['path_current_cases'] = temp_config['path_current_cases'].replace('test', 'train')
            temp_config['true_labels_json'] = temp_config['true_labels_json'].replace('test', 'train')

        with open(f'./{folder_name}/config_{i}.json', 'w') as f:
            json.dump(temp_config, f, indent = 4)
