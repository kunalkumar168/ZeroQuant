# ZeroQuant Model Compression examples

Examples in this folder are helpful to try out some features and models that take advantage of the DeepSpeed compression library.

A detailed tutorial for understanding and using DeepSpeed model compression features can be seen from here: https://www.deepspeed.ai/tutorials/model-compression/

For BERT :
```
python bert-main.py --task-name cola --quant-config quant_configs/bert_config.json
```

For RoBERTa :

```
python roberta-main.py --model-name roberta-base --task-name qnli --quant-config quant_configs/roberta_config.json
python roberta-main.py --model-name roberta-large --task-name qnli --quant-config quant_configs/roberta_config.json
```
or Run for all the task at once by -
```
bash roberta_run.bash
```