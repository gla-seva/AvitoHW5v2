from typing import List
from math import log


class TfidfTransformer:
    @staticmethod
    def tf_transform(count_matrix: List[List[int]]) -> List[List[float]]:
        """
        Считаем нормированную частоту встречаемости слов в документах
        """
        def _normalize(vec: List[int]) -> List[float]:
            total = sum(vec)
            return [v / total for v in vec]

        tf_matrix = [_normalize(vec) for vec in count_matrix]
        return tf_matrix

    @staticmethod
    def idf_transform(count_matrix: List[List[int]]) -> List[float]:
        """
        Считаем нормированную частоту встречаемости слов среди документов
        """
        def _counter_positive(vec: List[int]) -> int:
            return len([v for v in vec if v > 0])

        vocab_size = len(count_matrix[0]) if len(count_matrix) else 0
        df_vector = [_counter_positive([vec[idx] for vec in count_matrix])
                     for idx in range(vocab_size)]

        total_docs = len(count_matrix)
        return [1 + log((total_docs + 1) / (df + 1)) for df in df_vector]
