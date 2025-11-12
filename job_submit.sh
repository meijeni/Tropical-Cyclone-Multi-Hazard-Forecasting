#!/bin/bash
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! Name of the job:
#SBATCH -J gpujob
#! Which project to charge:
#SBATCH -A CAULFIELD-SL3-GPU
#SBATCH -p ampere
#! How many nodes do you want?
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#! How many (MPI) tasks in total? (must be <= nodes*56)
#! cclake nodes have 56 CPUs (core) each and 3410MB of memory per cpu
#SBATCH --ntasks=1
#! How much wallclock running time?
#SBATCH --time=00:59:00
#! Additional options (keep as is)
#SBATCH --mail-type=NONE
#SBATCH --no-requeue

#! Notes:
#! Charging is determined by cpu number*walltime.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*56 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.
#! Each task is allocated 1 CPU by default, and each CPU is allocated 3420 MiB
#! of memory. If this is insufficient, also specify
#! --cpus-per-task and/or --mem (the latter specifies MiB per node).

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

#! ############################################################

#! Prepare required modules:
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-ccl              # REQUIRED - loads the basic environment

#! Conda environment
source /home/${USER}/.bashrc
conda activate my_env

pip list

#! Full path to application executable: 
application="python3 model/stormsurge_resnet_opt_shap.py"

#! Run options for the application:
options=""

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # sets workdir to the directory in which sbatch is run.

#! MPI attributes
export OMP_NUM_THREADS=1
np=$[${numnodes}*${mpi_tasks_per_node}] # do not touch

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets

###############################################################
##################### MAIN EXECUTABLE #########################
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

# Get Job's ID
JOBID=$SLURM_JOB_ID

# Create logging directory
mkdir -p logs
mkdir -p logs/${JOBID}/

# Save output logs to .out and error logs in .err within logging dir
exec 1> logs/${JOBID}/slurm-${JOBID}.out
exec 2> logs/${JOBID}/slurm-${JOBID}.err

# Run main.py
CMD="$application $options > logs/$JOBID/output.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > "logs/$JOBID/machine.file.$JOBID"
        echo -e "\nNodes allocated:\n================"
        echo `cat "logs/$JOBID/machine.file.$JOBID" | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
