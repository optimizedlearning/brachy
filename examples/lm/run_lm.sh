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

cd /projectnb/aclab/cutkosky/hax
module load python3 pytorch cuda/11.6 jax
TEMPFILE=`mktemp -d`
python -m venv $TEMPFILE
source $TEMPFILE/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python examples/lm/train_lm.py --wandb $@
