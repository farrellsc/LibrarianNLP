> Resource list:
> [[DrQA Code](https://github.com/farrellsc/DrQA#trained-models-and-data)|
> [DrQA Paper](https://arxiv.org/pdf/1704.00051.pdf)]

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

**TODO**:  
1. DrQA-scripts-reader-preprocess.py: process of raw data to coreNLP
2. DrQA-scripts-reader-predict.py: process of prediction
3. Use DrQA retriver & tokenizer on LibNLP for now

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
│       ├── TokenDictionary.py
│       └── utils.py
├── config
├── data
├── test
└── scripts
```

![LibNlp Code Structure](/img/LibNlp.png)  

---
