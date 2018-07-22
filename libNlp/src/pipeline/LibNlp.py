class LibNlp:
    """
    General pipeline of this project.
    """

    def __init__(self, batch_size: int = 128, tokenizer = None, ranker_config = None, reader_model = None):
        """
        Initialize the pipeline.

        :param batch_size: batch size when processing paragraphs.
        :param tokenizer: string option to specify tokenizer used on docs.
        :param ranker_config: config for ranker.
        :param reader_model: model file from which to load the DocReader.
        """
        self.tokenizer = None
        self.ranker = None
        self.reader = None
        raise NotImplementedError

    def process(self, query, top_n=1, n_docs=5):
        """
        Run queries (call process_batch).

        :param query: queries
        :param top_n: number of top answers to return from reader per batch element.
        :param n_docs: number of docs to return from retriever per batch element.
        :return: all predictions
        """
        predictions = self.process_batch([query], top_n, n_docs)
        return predictions[0]

    def process_batch(self, query, top_n=1, n_docs=5):
        """
        Run a batch of queries (more efficient).

        :param query: queries
        :param top_n: number of top answers the reader should return
        :param n_docs: number of docs the retriever should return
        :return: all predictions
        """
        # zzhuang: 1. use data retriever to retrieve top k ranking articles
        # zzhuang: 2. flattening retrieved articles
        # zzhuang: 3. tokenizing flattened sentences
        # zzhuang: 4. structure Question tokens and Doc tokens into minimal units
        # zzhuang: 5. split above examples with pytorch data iterator, then use pre-trained reader model to predict
        # zzhuang: 6. find top paragraph predictions and return
        raise NotImplementedError
