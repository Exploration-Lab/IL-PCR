import os, sys, json

if __name__ == '__main__':
    assert(len(sys.argv) > 1)
    folder_name = sys.argv[1]
    assert 'test' in folder_name.lower() and 'train' not in folder_name.lower(), f"Name : {folder_name} is wrong. Must contain just the keyword \"test\""

    # make the train folder
    train_folder = folder_name.replace('test', 'train')
    os.makedirs(train_folder, exist_ok=False)

    # for each config in test folder convert to train config and save
    for config_path in os.listdir(folder_name):
        with open(folder_name + f'/{config_path}', 'r') as f:
            config = json.load(f)

        for key, value in config.items():
            if type(value) == str:
                config[key] = value.replace('test', 'train')
        
        out_path = train_folder + f'/{config_path}'
        with open(out_path, 'w+') as f:
            json.dump(config, f, indent = 4)
