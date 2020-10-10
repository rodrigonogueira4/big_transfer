# wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz

python3 -m bit_pytorch.train --name covid_`date +%F_%H%M%S` --model BiT-M-R50x1 --logdir /scratch/rfn216/temp/bit_logs --dataset covid  --datadir ./data  --batch 128 --batch_split 8 --base_lr 0.001 --eval_every 1

