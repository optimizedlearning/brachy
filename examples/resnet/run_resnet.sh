#!/bin/bash -l

# Request 4 CPUs
#$ -pe omp 8

# Request 1 GPU
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -l gpu_memory=14G

#specify a project (probably not necessary, so currently off)
##     $ -P aclab

#merge the error and output
#$ -j y

#send email at the end
#$ -m e

# assuming we are in runs/ here:

module load python3 pytorch tensorflow cuda/11.2 cudnn/8.1.1
source /projectnb/aclab/cutkosky/gpujaxenv/bin/activate
pip install -r requirements.txt
python examples/train_resnet.py
