# coding=utf-8
from collections import Counter
from typing import List, Iterable
import re


class CountVectorizer:
    """
    Класс создан для посторения матрицы с частотой встречаемости слов по корпусу текстов
    """
    def __init__(self, lowercase: bool = True, token_pattern=r"(?u)\b\w\w+\b"):
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self._vocabulary = None

    def _preprocessing(self, texts: Iterable[str]) -> Iterable[str]:
        """
        Приведение слов документов к нижнему регистру
        :param texts: итерируемый объект с документами
        :return: итерируемый объект документов после обработки
        """
        if self.lowercase:
            return (str.lower(text) for text in texts)
        else:
            return texts

    def _tokenization(self, texts: Iterable[str]) -> Iterable[List[str]]:
        """
        Разбивает каждый из документов на отдельные токены
        :param texts:  итерируемый объект с документами
        :return: итерируемый объект со списками токенов
        """
        return (re.findall(self.token_pattern, text) for text in texts)

    def _extract_vocab(self, texts: Iterable[List[str]]) -> None:
        """
        Формирует общий словарь токенов по корпусу токенизированных документов
        :param texts: корпус токенизированных документов
        """
        counter = Counter()
        for text in texts:
            counter.update(text)
        self._vocabulary = list(counter)

    def _terms_counter(self, text: List[str]) -> List[int]:
        """
        Считает количество вхождений в предложение для каждого из слов словаря
        :param text: список токенов
        :return: вектор числа вхождений
        """
        counter = Counter(text)
        return [counter[word] for word in self._vocabulary]

    def fit(self, x: Iterable) -> 'CountVectorizer':
        """
        Проводит препроцессинг и токенизацию и формирует словарь
        :param x: корпус додументов
        :return: объект класс
        """
        x_processed = self._preprocessing(x)
        x_tokenized = self._tokenization(x_processed)
        self._extract_vocab(x_tokenized)
        return self

    def transform(self, x: Iterable) -> List[List[int]]:
        """
        Проводит препроцессинг и токенизацию и строит матрицу
        числа вхождений слов в каждый из документ корпуса
        :param x: корпус додументов
        :return: матрица числа вхождений
        """
        if not self._vocabulary:
            raise AttributeError('Нодопустим вызов transform без fit')
        x_processed = self._preprocessing(x)
        x_tokenized = self._tokenization(x_processed)
        return [self._terms_counter(text) for text in x_tokenized]

    def fit_transform(self, x: Iterable) -> List[List[int]]:
        """
        Последовательно вызывает методы fit и transform
        :param x: корпус додументов
        :return: матрица числа вхождений
        """
        return self.fit(x).transform(x)

    def get_feature_names(self) -> List[str]:
        """
        Возвращает список токенов всего корпуса
        :return: Список токенов
        """
        if not self._vocabulary:
            raise AttributeError('Мы еще не создали список фичей. Вызовите fit')
        return self._vocabulary
