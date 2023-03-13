run_experiment() {
    exp_name=$1
    for input in "${@:2}"; do
        config_name=./config_files/$exp_name/config_$input.json # fill your config pattern here
        log_path=./logs/$exp_name\_$input.txt
        if [[ -f $config_name ]]; then
            python3 -u ./run_script.py $config_name 1>$log_path 2>&1 & 
        else 
            echo "config_file DOES NOT exists : $config_name"
        fi
    done
}

mkdir -p logs
mkdir -p logs/sentence_removed

# for the ILPCR corpus
run_experiment configs_ik_train 1 2
run_experiment configs_ik_test 1 2
run_experiment configs_ik_train_atomic 1 2 3
run_experiment configs_ik_test_atomic 1 2 3
run_experiment configs_ik_train_events 1 2 3 4 5 
run_experiment configs_ik_test_events 1 2 3 4 5
run_experiment configs_ik_train_iouf 1 2 3 4 5
run_experiment configs_ik_test_iouf 1 2 3 4 5
run_experiment configs_ik_train_RR 1 2 3 4 5
run_experiment configs_ik_test_RR 1 2 3 4 5

# for citation sentence removed ILPCR
run_experiment sentence_removed/configs_ik_train 1 2
run_experiment sentence_removed/configs_ik_test 1 2
run_experiment sentence_removed/configs_ik_train_atomic 1 2 3
run_experiment sentence_removed/configs_ik_test_atomic 1 2 3
run_experiment sentence_removed/configs_ik_train_events 1 2 3 4 5 
run_experiment sentence_removed/configs_ik_test_events 1 2 3 4 5
run_experiment sentence_removed/configs_ik_train_iouf 1 2 3 4 5
run_experiment sentence_removed/configs_ik_test_iouf 1 2 3 4 5
run_experiment sentence_removed/configs_ik_train_RR 1 2 3 4 5
run_experiment sentence_removed/configs_ik_test_RR 1 2 3 4 5

# for COLIEE21
run_experiment configs_coliee21 1 2 3 4 5 6 7 8 9 10
run_experiment configs_coliee21_train_atomic 1 2 3
run_experiment configs_coliee21_test_atomic 1 2 3
run_experiment configs_coliee21_train_events 1 2 3 4 5 
run_experiment configs_coliee21_test_events 1 2 3 4 5
run_experiment configs_coliee21_train_iouf 1 2 3 4 5
run_experiment configs_coliee21_test_iouf 1 2 3 4 5
run_experiment configs_coliee21_train_RR 1 2 3 4 5
run_experiment configs_coliee21_test_RR 1 2 3 4 5
