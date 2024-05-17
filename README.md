# ZeroQuant Implementation

This is an implementation of the paper [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/abs/2206.01861)

For setting up the environment we recommend using a conda environment or virtual environment. For installing the required packages, run - 
```
pip install -r requirements.txt
```
Our implementation supports both 8 and 4 bit weight and activation quantizations. You can also specify different weight precisions for different layers in the model. Using this you can easily peform mixed precision weight quantization.
However, the layer must be an instance of [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) or [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html).

You can find the quant configurations in the [quant_configs](quant_configs) folder.

For quantizing a BERT model and evaluating on CoLA task of the [GLUE](https://gluebenchmark.com/) benchmark:
```
python bert-main.py --task-name cola --quant-config quant_configs/bert_config.json
```

Similarly, for running RoBERTa run:

```
python roberta-main.py --model-name roberta-base --task-name qnli --quant-config quant_configs/roberta_config.json
python roberta-main.py --model-name roberta-large --task-name qnli --quant-config quant_configs/roberta_config.json
```
or Run for all the task at once by -
```
bash roberta_run.bash
```


For quantizing GPT-2 XL model and evaluating perplexity on [Wikitext-2](https://huggingface.co/datasets/wikitext) dataset, run:

```
python decoder-perplexity.py --dataset  wikitext --dataset-config wikitext-2-raw-v1 --model gpt2-xl --tokenizer gpt2-xl  --batch-size 4 --quant-config quant_configs/gpt2_xl_config_W8A8.json
```
Using decoder-perplexity.py, you can quantize any decoder models, you just need to configure the quant configuration for that model.
