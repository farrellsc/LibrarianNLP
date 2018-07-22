> Resource list:
> [[DrQA Code](https://github.com/farrellsc/DrQA#trained-models-and-data)|
> [DrQA Paper](https://arxiv.org/pdf/1704.00051.pdf)]

# 1. Code Reading

Core Utils to Read:
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

![DrQA Code Structure](/img/DrQA.jpg)

## 1.1 Classs DrQA
> drqa.pipeline.drqa

### Critical Class Members



### Critical Methods
#### process
calls process_batch



# 2. Design

