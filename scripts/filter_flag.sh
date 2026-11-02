N_CPU=6
DUMP="fineweb2_cmn"
MAIN_OUTPUT_PATH="data/filterflag"
INPUT_DIR="data/fineweb2/data/cmn_Hani/train"

python fineweb2/transform.py ${INPUT_DIR} ${DUMP} ${MAIN_OUTPUT_PATH} ${N_CPU}  # transform into acceptable format
python fineweb2/filter_lang.py ${DUMP} ${MAIN_OUTPUT_PATH} ${N_CPU}
python fineweb2/zhtwplus.py ${DUMP} ${MAIN_OUTPUT_PATH} 1