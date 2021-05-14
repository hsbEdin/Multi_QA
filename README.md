# Multi Conversational Question Answering Model

This is a multi conversational question answering. 
We integrate the historical turn information into the vector, and get the extracted and generated answers by discrimination.

```

### Run

1. Download the `BERT-Base/BERT-Large Uncased` model [here](https://github.com/google-research/bert).
2. Download the [CoQA](https://stanfordnlp.github.io/coqa/) [QuAC](http://quac.ai/) data.
3. Create a file folder named "Data" to store the dataset.
4. Run

```
python main.py conf \

```
Using bert large model should give better results, but it's more demanding in CUDA memory.

5. After training, the best result is stored in the output directory.

### Scripts

* `main.py`. Entry code.
* `ConvQA_CN_NetTrainer.py`. Generate batches.
* `CoQAUtils.py/QuACUtils.py`. Utility functions.
* `ConvQA_CN_Net.py`. Our models.
* `evaluate.py.py`. Official evaluation script for CoQA.

Most other files are for BERT, AdamW and Ranger.


### Environment

Tested with Python 3.6.7 and Torch 1.5.0
