for i in 10 100; do
    for j in 1 2 3 4 5; do

    start_time=$(date +%s)

    python3 cli.py \
    --method pet \
    --pattern_ids 1 6 \
    --data_dir ../data/ag/$i/$j/ \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name agnews \
    --output_dir results/ag/$i/$j/ \
    --overwrite_output_dir \
    --do_train \
    --do_eval  \
    --pet_per_gpu_eval_batch_size 1 \
    --pet_per_gpu_train_batch_size 1 \
    --pet_max_seq_length 512 \
    --pet_num_train_epochs 10 \
    --pet_repetitions 5 \
    --eval_set 'test'

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Log the duration
    echo "i:$i, j:$j $duration seconds" >> ag_news_time_log.txt

    done
done
