from count_vectorizer import CountVectorizer
from tfidf_transformer import TfidfTransformer
from typing import List, Iterable


class TfidfVectorizer(CountVectorizer, TfidfTransformer):
    def __init__(self, lowercase: bool = True, token_pattern: str = r"(?u)\b\w\w+\b"):
        super().__init__(lowercase, token_pattern)

    def transform(self, x: Iterable) -> List[List[float]]:
        """
        Строит матрицу df-idf
        """
        count_matrix = super().transform(x)
        tf_matrix = self.tf_transform(count_matrix)
        idf_vector = self.idf_transform(count_matrix)

        tfidf_matrix = [[tf * idf for tf, idf in zip(tf_vector, idf_vector)]
                        for tf_vector in tf_matrix]
        return tfidf_matrix
