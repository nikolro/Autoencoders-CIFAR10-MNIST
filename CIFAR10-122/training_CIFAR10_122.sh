#!/bin/bash

# Ensure the script exits if any command fails
set -e
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
MAIL_USER="nikol.ro@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
# Define the output directory
OUTPUT_DIR="/home/nikol.ro/DL/project/code/out"
NOTEBOOK_PATH="/home/nikol.ro/DL/project/code/CIFAR10-122/CIFAR10_122.ipynb"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Function to run the notebook
run_notebook(){
  NOTEBOOK_NAME="CIFAR10_122.ipynb"

  sbatch \
    -N $NUM_NODES \
    -c $NUM_CORES \
    --gres=gpu:$NUM_GPUS \
    --job-name "notebook_run" \
    --mail-user $MAIL_USER \
    --mail-type $MAIL_TYPE \
    --time=04:00:00 \
    -o "${OUTPUT_DIR}/notebook_run_CIFAR10_122.out" \
    <<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB 'notebook_run' of '${NOTEBOOK_NAME}' STARTING ***"

# Setup the conda env
source \$HOME/DL/miniconda3/etc/profile.d/conda.sh
conda activate cs236781-hw

# Run the notebook 
python ../training.py run-nb ${NOTEBOOK_PATH}

echo "*** SLURM BATCH JOB 'notebook_run' DONE ***"
EOF
}

# Call the function to run the notebook
run_notebook