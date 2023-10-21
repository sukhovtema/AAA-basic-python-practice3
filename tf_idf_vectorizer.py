from count_vectorizer import CountVectorizer
from tf_idf_transformer import TfidfTransformer


class TfidfVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self.tf_idf_transformer = TfidfTransformer()

    def fit_transform(self, corpus: list[str]) -> list[list]:
        """Transform the text into a feature matrix using CountVectorizer.
        Get TF-IDF matrix using TfidfTransformer."""
        corpus_matrix = super().fit_transform(corpus)
        return self.tf_idf_transformer.fit_transform(corpus_matrix)
