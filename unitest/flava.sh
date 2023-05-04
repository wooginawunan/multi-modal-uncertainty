for DATA in "hateful-meme-dataset" "food101"
do
    for TYPE in "Vanilla" "MIMO-shuffle-instance" "MultiHead"
    do
        python train.py --use_gpu --device 0 --verbose \
        --save_path $RESULTS_DIR/unitest \
        --lr 3e-5 --batch_size 4 --n_epochs 1 \
        --dataset $DATA --sample_size 200 \
        --framework flava \
        --model_type $TYPE \
        > unitest/out/flava_$DATA 2>unitest/error/flava_$DATA

        python train.py --use_gpu --device 0 --verbose \
        --save_path $RESULTS_DIR/unitest \
        --lr 3e-5 --batch_size 4 --n_epochs 1 \
        --dataset $DATA --sample_size 200 \
        --framework flava \
        --model_type $TYPE --clstoken \
        > unitest/out/clstoken_$DATA 2>unitest/error/clstoken_$DATA

        python train.py --use_gpu --device 0 --verbose \
        --save_path $RESULTS_DIR/unitest \
        --lr 3e-5 --batch_size 4 --n_epochs 1 \
        --dataset $DATA --sample_size 200 \
        --framework flava \
        --model_type $TYPE --avg_pool \
        > unitest/out/avg_pool_$DATA 2>unitest/error/avg_pool_$DATA
    done
done