#!/bin/bash -l
#PBS -N derecho_regional_mom6_emulation
#PBS -l select=1:ncpus=12
#PBS -l walltime=00:30:00
#PBS -A P93300012
#PBS -q main
#PBS -j oe
#PBS -m bae 

module purge
module load ncarenv/24.12
module reset
module load gcc craype cray-mpich cuda cudnn conda
module load mkl # necessary for pytorch

export LSCRATCH=/glade/derecho/scratch/ajanney/
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=hsn
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export NCCL_NCHANNELS_PER_NET_PEER=4
export MPICH_RDMA_ENABLED_CUDA=1
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PBH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
# logger.info the results
echo "Number of nodes: 1"
echo "Number of GPUs per node: 4"
echo "Total number of GPUs: 4"
# Log in to WandB if needed
# wandb login 02d2b1af00b5df901cb2bee071872de774781520
# Launch MPIs
nodes=( $( cat $PBS_NODEFILE ) )
echo nodes: $nodes

# Find headnode's IP:
head_node=${nodes[0]}
head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')
MASTER_ADDR=$head_node_ip MASTER_PORT=1234 mpiexec -n 32 --ppn 4 --cpu-bind none python /glade/work/schreck/repos/miles-credit/applications/train.py  -c samudra.yml --backend nccl

conda activate credit-derecho

cd /glade/work/ajanney/miles-credit/credit/trainers

export PYTHONUNBUFFERED=1
python trainer_rMOM6.py