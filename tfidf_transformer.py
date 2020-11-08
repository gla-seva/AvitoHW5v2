from typing import List
from math import log


class TfidfTransformer:
    @staticmethod
    def _normalize(vec: List[int]) -> List[float]:
        total = sum(vec)
        return [v / total for v in vec]

    @classmethod
    def tf_transform(cls, count_matrix: List[List[int]]) -> List[List[float]]:
        """
        Считаем нормированную частоту встречаемости слов в документах
        """
        tf_matrix = [cls._normalize(vec) for vec in count_matrix]
        return tf_matrix

    @staticmethod
    def _counter_positive(vec: List[int]) -> int:
        return len([v for v in vec if v > 0])

    @classmethod
    def idf_transform(cls, count_matrix: List[List[int]]) -> List[float]:
        """
        Считаем нормированную частоту встречаемости слов среди документов
        """
        vocab_size = len(count_matrix[0]) if len(count_matrix) else 0
        df_vector = [cls._counter_positive([vec[idx] for vec in count_matrix])
                     for idx in range(vocab_size)]

        total_docs = len(count_matrix)
        return [1 + log((total_docs + 1) / (df + 1)) for df in df_vector]

    def fit_transform(self, count_matrix: List[List[int]]) -> List[List[float]]:
        """
        Строит матрицу df-idf
        """
        tf_matrix = self.tf_transform(count_matrix)
        idf_vector = self.idf_transform(count_matrix)

        tfidf_matrix = [[tf * idf for tf, idf in zip(tf_vector, idf_vector)]
                        for tf_vector in tf_matrix]
        return tfidf_matrix
