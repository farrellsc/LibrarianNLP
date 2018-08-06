class FeatureDict(dict):
    def __init__(
            self,
            examples,
            use_qemb=False,
            use_in_question=False,
            use_pos=False,
            use_ner=False,
            use_lemma=False,
            use_tf=False
    ):
        self.use_qemb = use_qemb
        self.use_in_question = use_in_question
        self.use_pos = use_pos
        self.use_ner = use_ner
        self.use_lemma = use_lemma
        self.use_tf = use_tf
        self.feature_dict = self.build_feature_dict(examples)

    def __getitem__(self, key):
        return self.feature_dict.get(key, False)

    def __len__(self):
        return len(self.feature_dict)

    def build_feature_dict(self, examples):
        """Index features (one hot) from fields in examples and options."""

        def _insert(feature):
            if feature not in feature_dict:
                feature_dict[feature] = len(feature_dict)

        feature_dict = {}

        # Exact match features
        if self.use_in_question:
            _insert('in_question')
            _insert('in_question_uncased')
            if self.use_lemma:
                _insert('in_question_lemma')

        # Part of speech tag features
        if self.use_pos:
            for ex in examples.examples:
                for w in ex['pos']:
                    _insert('pos=%s' % w)

        # Named entity tag features
        if self.use_ner:
            for ex in examples.examples:
                for w in ex['ner']:
                    _insert('ner=%s' % w)

        # Term frequency feature
        if self.use_tf:
            _insert('tf')
        return feature_dict
