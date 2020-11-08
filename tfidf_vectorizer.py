from count_vectorizer import CountVectorizer
from tfidf_transformer import TfidfTransformer
from typing import Iterable, List


class TfidfVectorizer(CountVectorizer):
    def __init__(self, lowercase: bool = True, token_pattern: str = r"(?u)\b\w\w+\b"):
        super().__init__(lowercase, token_pattern)
        self._tfidf = TfidfTransformer()

    def transform(self, x: Iterable) -> List[List[float]]:
        """
        Трансформер последовательно применяет сначала из CountVectorizer затем TfidfTransformer
        """
        count_matrix = super().transform(x)
        tfidf_matrix = self._tfidf.fit_transform(count_matrix)
        return tfidf_matrix
