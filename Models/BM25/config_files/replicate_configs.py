import os, sys, json

if __name__ == '__main__':
    assert(len(sys.argv) > 1)
    folder_name = sys.argv[1]
    assert(len(os.listdir(folder_name)) == 1)
    with open(f'./{folder_name}/config_1.json', 'r') as f:
        config = json.load(f)
        assert(config["n_gram"] == 1)
    for i in range(2,6):
        temp_config = config
        temp_config["n_gram"] = i
        with open(f'./{folder_name}/config_{i}.json', 'w') as f:
            json.dump(temp_config, f, indent = 4)
