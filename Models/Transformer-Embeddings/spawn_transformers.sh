# it is not recommended to run all the experiments at once as your GPU will run out of memory.

run_experiment() {  
    exp_name=$1
    for input in "${@:2}"; do
        export CUDA_VISIBLE_DEVICES=$(((input-1)%4))    # running on one of 4 gpus, change this for your system
        config_name=./config_files/$exp_name/config_$input.json # fill your config pattern here
        log_path=./logs/$exp_name\_$input.txt
        last_slash_index=${log_path%/*}
        log_path="${log_path/$last_slash_index\//${last_slash_index}_}"
        if [[ -f $config_name ]]; then
            python3 -u ./transformer_score.py $config_name 1>$log_path 2>&1 & 
        else 
            echo "config_file DOES NOT exists : $config_name"
        fi
    done
}

mkdir -p logs

# ILPCR experiments
run_experiment configs_ILPCR/bert 1 2 3 4
run_experiment configs_ILPCR/bert_finetuned 1 2 3 4
run_experiment configs_ILPCR/distilbert 1 2 3 4
run_experiment configs_ILPCR/distilbert_finetuned 1 2 3 4
run_experiment configs_ILPCR/InCaseLawBERT 1 2 3 4
run_experiment configs_ILPCR/InLegalBERT 1 2 3 4

# citation-removed sentences ILPCR experiments
# run_experiment configs_ILPCR_citation_removed/bert 1 2 3 4
# run_experiment configs_ILPCR_citation_removed/bert_finetuned 1 2 3 4
# run_experiment configs_ILPCR_citation_removed/distilbert 1 2 3 4
# run_experiment configs_ILPCR_citation_removed/distilbert_finetuned 1 2 3 4
# run_experiment configs_ILPCR_citation_removed/InCaseLawBERT 1 2 3 4
# run_experiment configs_ILPCR_citation_removed/InLegalBERT 1 2 3 4

# COLIEE experiments 
# run_experiment configs_COLIEE/bert 1 2 3 4
# run_experiment configs_COLIEE/bert_finetuned 1 2 3 4
# run_experiment configs_COLIEE/distilbert 1 2 3 4
# run_experiment configs_COLIEE/distilbert_finetuned 1 2 3 4
# run_experiment configs_COLIEE/InCaseLawBERT 1 2 3 4
# run_experiment configs_COLIEE/InLegalBERT 1 2 3 4
