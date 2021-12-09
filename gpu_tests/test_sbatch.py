#!/home/miranda9/.conda/envs/automl-meta-learning/bin/python3.7
#SBATCH --job-name="miranda9job"
#SBATCH --output="experiment_output_job.%j.%N.out"
#SBATCH --error="experiment_output_job.%j.%N.err"
#SBATCH --export=ALL
#SBATCH --mail-user=brando.science@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00
#SBATCH --partition=csl

print('Hi')
