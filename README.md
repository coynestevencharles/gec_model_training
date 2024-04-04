# gec_model_training

Code for training [Buntan/gec-t5-v1_1-small](https://huggingface.co/Buntan/gec-t5-v1_1-small), a small T5v1.1 model trained on the cLang-8 dataset following the process described in [A Simple Recipe for Multilingual Grammatical Error Correction](https://arxiv.org/abs/2106.03830).

The data was obtained from Google's cLang-8 repository [here](https://github.com/google-research-datasets/clang8).

The dataset was generated with the `--tokenize_text='False'` option to produce `clang8_source_target_en.tsv`, a file of sentence pairs suitable for T5 training.

The data was then split into 95% training and 5% validation and converted into a Hugging Face Dataset object (see `data_preparation.py`).

The training code can be found in `train_script.py`. The hyperparameters used are below:

|Param|Value|
|:--|:--|
|epochs|3|
|batch size|32|
|learning rate|2e-5|
|lr scheduler type|linear|
|optimizer|AdamW|
|max length|128|
|mixed precision|fp16|
|gradient accumulation|3|
