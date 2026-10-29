N_SHARD=6
N_CPU=16
INPUT_DIR="data/fineweb2/data/cmn_Hani/train"
DUMP="fineweb2_cmn"
MAIN_OUTPUT_PATH="data/parsed"
DATA_OUTPUT_PATH="data/fineweb2"

python fineweb2/download.py ${DATA_OUTPUT_PATH} ${N_SHARD}                      # download fineweb2 from HF
python fineweb2/transform.py ${INPUT_DIR} ${DUMP} ${MAIN_OUTPUT_PATH} ${N_CPU}  # transform into acceptable format
