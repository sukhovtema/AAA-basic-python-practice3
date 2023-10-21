from math import log


class TfidfTransformer:
    @staticmethod
    def tf_transform(corpus_matrix: list) -> list[list]:
        """Transform the Term Frequency (TF) matrix within a corpus."""
        return [[el / sum(counts) for el in counts]
                for counts in corpus_matrix]

    @staticmethod
    def idf_transform(corpus_matrix: list) -> list[float]:
        """Transform the Inverse Document Frequency (IDF) vector."""
        X = []
        len_corpus_matrix = len(corpus_matrix) + 1

        for i in range(len(corpus_matrix[0])):
            word_docs = 1
            for corpus in corpus_matrix:
                if corpus[i] > 0:
                    word_docs += 1

            X.append(log(len_corpus_matrix / word_docs) + 1)

        return X

    def fit_transform(self, corpus_matrix: list[float]) -> list[list]:
        """Calculate TF-IDF from the Term Frequency matrix within a corpus."""
        tf_matrix = self.tf_transform(corpus_matrix)
        idf_vector = self.idf_transform(corpus_matrix)

        return [[tf_vector[i] * idf_vector[i] for i in range(len(tf_vector))]
                for tf_vector in tf_matrix]
