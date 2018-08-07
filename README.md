> Resource list:
> [[DrQA Code](https://github.com/farrellsc/DrQA#trained-models-and-data)|
> [DrQA Paper](https://arxiv.org/pdf/1704.00051.pdf)]

---

# 0. Current Objective

```bash
python scripts/train.py --embedding-file glove.840B.300d.txt
```

**DONE**:
Use `train_noLog.py` as testing baseline.
1. Data Section is finished & tested.
2. Reader Section is rounded-up, core functions in progress.

**TODO**:  
1. Now
    1. Training Section: Reader.update -> Model.forward -> Network.forward;  
    2. Predicting Section: Reader.predict  
    3. add detailed description to every method and class; add TYPE to parameters and returns  
    4. Retriever & Tokenizer: temporarily use sections from DrQA  
2. Future  
    1. Use DrQA retriver & tokenizer on LibNLP for now  
    2. DrQA-scripts-reader-preprocess.py: process of raw data to coreNLP  
    3. DrQA-scripts-reader-predict.py: process of prediction  

---

# 1. DrQA

Core sections to Read:
```
DrQA root
├── drqa
│   ├── pipeline
│   │   └── drqa.py
│   ├── reader
│   |   ├── utils.py
│   |   ├── rnn_reader.py
│   |   ├── layers.py
|   |   └── model.py
│   └── tokenizer
└── scripts
    ├── pipeline
    │   └── interactive.py
    ├── convert
    │   └── squad.py
    └── reader
        ├── preprocess.py
        └── train.py
```

![DrQA Code Structure](/img/DrQA.png)

Current Code Structure is based on "DrQA-scripts-reader-train.py".  

---

# 2. LibNLP Design
Core Utils to Read:
```
LibNLP root
├── src
│   ├── pipeline
│   │   └── LibNlp.py
│   ├── data
│   │   └── LibDataLoader.py
│   ├── reader
│   |   ├── Reader.py
|   |   ├── Model.py
|   |   └── networks
│   |       ├── Network.py
│   |       ├── BilinearSeqAttn.py
│   |       └── StackedBRNN.py
│   ├── retriever
│   ├── tokenizer
│   └── utils
│       ├── AverageMeter.py
│       ├── Timer.py
│       ├── TokenDictionary.py
│       ├── Param.py
│       ├── register.py
│       └── utils.py
├── config
├── data
├── test
└── scripts
    ├── train_noLog.py
    └── train.py
```

![LibNlp Code Structure](/img/LibNlp.png)  

---
