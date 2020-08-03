#!/bin/bash
#$ -cwd
#$ -pe onenode 6
#$ -l m_mem_free=8G
python3 marketVolatility.py