for DATA in "hateful-meme-dataset" "food101"
do
    python train.py --use_gpu --device 0 --verbose \
    --save_path $RESULTS_DIR/unitest \
    --lr 3e-5 --batch_size 4 --n_epochs 1 \
    --dataset $DATA --sample_size 200 \
    --framework vilt --gradient_accumulation_steps 10 \
    > unitest/out/vilt_$DATA 2>unitest/error/vilt_$DATA

done

