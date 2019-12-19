#!/bin/bash
CORPUS_RAW=""
BERT_BASE_DIR=""
CORPUS_JSON_DIR=""

python pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \
                  --bert_model ${BERT_BASE_DIR}$ \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir ${CORPUS_JSON_DIR}$