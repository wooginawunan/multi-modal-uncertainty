
python train.py --use_gpu --device 0 --verbose \
--save_path $RESULTS_DIR/unitest \
--lr 3e-5 --batch_size 4 --n_epochs 1 \
--dataset food101 --sample_size 200 \
--framework mmbt --gradient_accumulation_steps 10 \
> unitest/out/mmbt 2>unitest/error/mmbt
