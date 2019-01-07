## Generating SQL Queries from Natural Language

This repo provides extension to the implementation of [SQLNet repository](https://github.com/xiaojunxu/SQLNet) and added modules for using character embeddings along with word embeddings for training the bidirectional LSTMs.


The code is implemented in python2.7 using the [PyTorch 0.3.1](http://pytorch.org). For install other requirements execute requirements.txt as
```bash
pip install -r requirements.txt
```

We have installed and extracted the pretrained GLOVE embeddings which can be used directly for training and testing the model. We have added a class inside word_embedding.py called CharacterEmbedding which contains modules and components related to character embeddings.

### For training the model, we need to run "train.py" with the following commands:

### For training Seq2SQL use 
```bash
python train.py --baseline
```

### For training SQLNet with column attention use (1)
```bash
python train.py --ca
```
### For training SQLNet using trainable embeddings first execute the previous command(1) then execute
```bash
python train.py --ca --train_emb
```

### For training SQLNet using character and word embeddings(our implementation) use
```bash
python train_embed_char.py --ca
```

## For testing the model first execute any of the above train commands which trains and saves the best model, then execute the corresponding test command as follows:

### For testing Seq2SQL use 
```bash
python test.py --baseline
```

### For testing SQLNet with column attention use (1)
```bash
python test.py --ca
```

### For testing SQLNet using trainable embeddings first execute the previous command(1) then execute
```bash
python test.py --ca --train_emb
```

### For testing SQLNet using character and word embeddings(our implementation) use
```bash
python test_embed_char.py --ca
```
