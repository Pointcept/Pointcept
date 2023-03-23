#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
export PYTHONPATH=./
PYTHON=python

TEST_CODE=test.py

DATASET=s3dis
CONFIG="None"
EXP_NAME=debug
WEIGHT=model_best

while getopts "p:d:c:n:w:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=${EXP_DIR}/config.py

if [ "${CONFIG}" = "None" ]
then
    CONFIG_DIR=${EXP_DIR}/config.py
else
    CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
fi

echo " =========> RUN TASK <========="

#$PYTHON -u "$CODE_DIR"/tools/$TEST_CODE \
$PYTHON -u tools/$TEST_CODE \
  --config-file "$CONFIG_DIR" \
  --options save_path="$EXP_DIR" weight="${MODEL_DIR}"/"${WEIGHT}".pth
