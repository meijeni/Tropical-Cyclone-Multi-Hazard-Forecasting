#!/usr/bin/env bash
module load miniconda/3
conda create -y -n my_env python=3.10
conda init
echo "Please restart your HPC instance before activating your new conda environment (Jeni Alert!)"
