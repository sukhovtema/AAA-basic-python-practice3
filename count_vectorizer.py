class CountVectorizer:
    def __int__(self):
        self.vocabulary = {}
        self.feature_names = []

    def fit_transform(self, corpus: list[str]) -> list[list]:
        """
        Train the CountVectorizer on a corpus of text and
        transform the text into a feature matrix.
        """
        self.vocabulary = {}
        self.feature_names = []

        if not isinstance(corpus, list):
            raise TypeError('List corpus expected.')
        for text in corpus:
            if not isinstance(text, str):
                raise TypeError(f'String object expected but get {type(text)}')

        for text in corpus:
            for word in text.split():
                word = word.lower()
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)
                    self.feature_names.append(word)

        X = []
        for text in corpus:
            words_counts = [0] * len(self.feature_names)
            for word in text.split():
                word = word.lower()
                words_counts[self.vocabulary[word]] += 1
            X.append(words_counts)

        return X

    def get_feature_names(self) -> list[list]:
        """Get output feature names for transformation."""
        return self.feature_names
