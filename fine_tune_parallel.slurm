#!/bin/bash
#SBATCH --job-name=1e-7    # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=512G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-12              # job array with index values 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=ns5404@princeton.edu

# remember to change the email and the job name with the learning rate you're trying

module purge
module load anaconda3/2023.9
conda activate dna

python fine_tune.py --username=$1 --model_out_dir=$2 --results_out_dir=$3 --histone=$4
