TinyBERT
======== 
TinyBERT is 7.5x smaller and 9.4x faster on inference than BERT-base and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages. The overview of TinyBERT learning is illustrated as follows: 
<br />
<br />
![](./tinybert_overview.png)
<br />
<br />

更多关于模型的细节直接参考原始论文:<br>
[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)


Release Notes
=============
First version: 2019/11/26

Installation
============
Run command below to install the environment(**using python3**)
```bash
pip install -r requirements.txt
```

General Distillation(通用蒸馏)
====================
通用蒸馏过程，使用官方提供的预训练BERT-base版本(未微调)作为teacher 模型，再使用大规模的语料进行学习。
具体来说：<br>
通用蒸馏步骤使用English Wikipedia作为语料。通用蒸馏会耗时2天。
<br>
在通用蒸馏阶段使用本文提出的Transformer蒸馏法在上述语料上进行蒸馏，从而习得通用知识。此时的
student模型称为general TinyBERT，该模型能够为下游具体任务提供一个具有良好初始化的模型。
通用蒸馏由2个步骤组成：<br>
(1)语料格式化<br>
(2)运行Transformer蒸馏


Step 1: 运行 `pregenerate_training_data.py`以生成json格式的语料 


```
 
# ${BERT_BASE_DIR}$ includes the BERT-base teacher model.
 
python pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \ 
                  --bert_model ${BERT_BASE_DIR}$ \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir ${CORPUS_JSON_DIR}$ 
                             
```

Step 2: 运行 `general_distill.py`以进行通用蒸馏
```
 # ${STUDENT_CONFIG_DIR}$ includes the config file of student_model.
 
python general_distill.py --pregenerated_data ${CORPUS_JSON}$ \ 
                          --teacher_model ${BERT_BASE}$ \
                          --student_model ${STUDENT_CONFIG_DIR}$ \
                          --reduce_memory --do_lower_case \
                          --train_batch_size 256 \
                          --output_dir ${GENERAL_TINYBERT_DIR}$ 
```

官网也提供了通用蒸馏的结果，一共2个版本：<br>

=================1st version to reproduce our results in the paper ===========================

[General_TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj) 

[General_TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1wXWR00EHK-Eb7pbyw0VP234i2JTnjJ-x)

=================2nd version (2019/11/18) trained with more (book+wiki) and no `[MASK]` corpus =======

[General_TinyBERT_v2(4layer-312dim)](https://drive.google.com/open?id=1PhI73thKoLU2iliasJmlQXBav3v33-8z)

[General_TinyBERT_v2(6layer-768dim)](https://drive.google.com/open?id=1r2bmEsQe4jUBrzJknnNaBJQDgiRKmQjF)


Data Augmentation(数据增强)
=================
数据增强是为了扩展下游任务训练集。更多任务相关数据集，student模型的泛化效果会更好。文章中的数据增强
方案：采用预训练的BERT和GloVe词嵌入进行word级别的替换。

运行 `data_augmentation.py` 进行数据增强，增强后的结果 `train_aug.tsv` 存于`${GLUE_DIR/TASK_NAME}$`
```

python data_augmentation.py --pretrained_bert_model ${BERT_BASE_DIR}$ \
                            --glove_embs ${GLOVE_EMB}$ \
                            --glue_dir ${GLUE_DIR}$ \  
                            --task_name ${TASK_NAME}$

```
对于GLUE数据集需要预先下载数据集[GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://github.com/nyu-mll/GLUE-baselines)。
再将数据解包到`GLUE_DIR`目录。 任务名TASK_NAME可以是： CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE.

Task-specific Distillation(任务蒸馏)
==========================
在任务蒸馏阶段也是使用本文提出的Transformer蒸馏法进一步改进上述已经蒸馏出的通用TinyBERT。
任务蒸馏更侧重于面向具体任务知识的学习。 

任务蒸馏包括了2个步骤:<br> 
(1) 中间层蒸馏<br> 
(2) 预测层蒸馏

Step 1: 运行 `task_distill.py`以执行中间层蒸馏
```

# ${FT_BERT_BASE_DIR}$ contains the fine-tuned BERT-base model.

python task_distill.py --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${GENERAL_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \ 
                       --output_dir ${TMP_TINYBERT_DIR}$ \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 10 \
                       --aug_train \
                       --do_lower_case  
                         
```


Step 2: 运行 `task_distill.py` 以执行预测层蒸馏
```

python task_distill.py --pred_distill  \
                       --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${TMP_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${TINYBERT_DIR}$ \
                       --aug_train  \  
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --eval_step 100 \
                       --max_seq_length 128 \
                       --train_batch_size 32 
                       
```

TinyBERT官方提供了2个版本最终TinyBERT模型，分是4层-312维和6层-768维(GLUE上所有的任务)。
每个任务都对应一个TinyBERT模型。


[TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1_sCARNCgOZZFiWTSgNbE7viW_G5vIXYg) 

[TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1Vf0ZnMhtZFUE0XoD3hTXc6QtHwKr_PwS)


Evaluation
==========================
运行 `task_distill.py` 对各个任务进行评估：

```
${TINYBERT_DIR}$ includes the config file, student model and vocab file.

python task_distill.py --do_eval \
                       --student_model ${TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${OUTPUT_DIR}$ \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128  
                                   
```
