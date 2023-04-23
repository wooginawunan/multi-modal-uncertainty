#!/usr/bin/env bash
export PNAME="multimodal_env"
export ROOT="/gpfs/data/geraslab/Nan/multi_modal_uncertainty"

export PYTHONPATH=${PYTHONPATH}:${ROOT}

export RESULTS_DIR=${ROOT}/saves/hateful-meme
export DATA_DIR=/gpfs/data/geraslab/Nan/multi_modal/hateful-meme-dataset
#export DATA_DIR=${ROOT}/fashionMNIST
    
# Switches off importing out of environment packages
export PYTHONNOUSERSITE=1

# if [ ! -d "${DATA_DIR}" ]; then
#   echo "Creating ${DATA_DIR}"
#   mkdir -p ${DATA_DIR}
# fi

if [ ! -d "${RESULTS_DIR}" ]; then
  echo "Creating ${RESULTS_DIR}"
  mkdir -p ${RESULTS_DIR}
fi

echo "Welcome to MULTIMODAL PROJECT:)"
echo "rooted at ${ROOT}"
echo "...With PYTHONPATH: ${PYTHONPATH}"
echo "...With RESULTS_DIR: ${RESULTS_DIR}"
echo "...With DATA_DIR: ${DATA_DIR}"