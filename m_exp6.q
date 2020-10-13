#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=rfnCMBPTT
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=6
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=rodrigonogueira4@gmail.com
#SBATCH --mail-type=END,FAIL 
#SBATCH --partition=p40_4,v100_sxm2_4,v100_pci_2,p100_4


# wget https://xray-livia.s3.us-east-2.amazonaws.com/h5files.zip
# wget https://xray-livia.s3.us-east-2.amazonaws.com/xray_550_test.hdf5
# wget https://storage.googleapis.com/bit_models/BiT-M-R152x4.npz

RUNDIR=/scratch/rfn216/test/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

nvidia-smi >> $RUNDIR/out.log

module purge
source $HOME/mainenv/bin/activate

python -m bit_pytorch.train --name covid_`date +%F_%H%M%S` --model BiT-M-R101x3 --logdir /scratch/rfn216/temp/bit_logs --dataset covid  --datadir ./data/h5files  --batch 128 --batch_split 64 --base_lr 0.001 --eval_every 4  >> $RUNDIR/out.log 2>&1

exit
