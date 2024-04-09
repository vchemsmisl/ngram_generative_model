import nltk.tokenize as tkn
from nltk.data import load
from pymorphy2.tokenizers import simple_word_tokenize
import re
from tqdm import tqdm
from collections import Counter
from math import log2
import random


class NotExistingTokenizerError(Exception):
    """
    Raised when an irrelevant tokenization_mode is passed
    to the Tokenizer class' instance
    """


class Tokenizer:

    def __init__(self, tokenization_mode: str, register: str = 'both') -> None:
        self.tokenization_mode = tokenization_mode
        self.register = register

    def tokenize(self, text) -> list[str]:
        match self.tokenization_mode:
            case 'split':
                tokens = self._split_tokenizer(text)
            case 'pymorphy':
                tokens = self._pymorphy_tokenizer(text)
            case 'with_punctuation':
                tokens = self._with_punctuation_tokenizer(text)
            case 'with_selected_punctuation':
                tokens = self._with_selected_punctuation_tokenizer(text)
            case 'with_tag':
                tokens = self._with_tag_tokenizer(text)
            case _:
                raise NotExistingTokenizerError('invalid tokenization mode')

        if self.register == 'both':
            return tokens
        elif self.register == 'lower':
            return [token.lower() for token in tokens]

    @staticmethod
    def _split_tokenizer(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def _pymorphy_tokenizer(text: str) -> list[str]:
        return simple_word_tokenize(text)

    @staticmethod
    def _with_punctuation_tokenizer(text: str) -> list[str]:
        return tkn.WordPunctTokenizer().tokenize(text)

    @staticmethod
    def _with_selected_punctuation_tokenizer(text: str) -> list[str]:
        text = re.sub(r'[()«»\'\"]', '', text)
        return tkn.WordPunctTokenizer().tokenize(text)

    @staticmethod
    def _with_tag_tokenizer(text: str) -> list[str]:
        tk = load('tokenizers/punkt/russian.pickle')
        sents = tk.tokenize(text)
        sents_with_tags = [re.sub(r'[!.?]', '***', sent) for sent in sents]

        tokens = []
        for sentence in sents_with_tags:
            tokens.extend(tkn.WordPunctTokenizer().tokenize(sentence))
        return tokens


class NgramModel:

    def __init__(self, n: int, smoothing_algorithm=None) -> None:
        self.n = n
        self._smoothing_algorithm = smoothing_algorithm

        self._num_tokens = 0
        self._num_tokens_unique = 0
        self._ngrams_count = None
        self.ngrams_probabilities = None

    def _get_ngrams(self, tokens) -> list[tuple[str]]:
        self._num_tokens = len(tokens)
        self._num_tokens_unique = len(set(tokens))

        ngrams = []
        for i in tqdm(range(len(tokens))):
            for j in range(self.n + 1):

                ngram = tuple(tokens[i:i+j])
                if ngram and ngram not in ngrams:
                    ngrams.append(ngram)
                    del ngram

        return ngrams

    def _get_ngrams_count(self, ngrams: list[tuple[str]]) -> None:
        self._ngrams_count = dict(Counter(ngrams))

    def _get_ngrams_probabilities(self, ngrams: list[tuple[str]]) -> dict[tuple[str], float]:
        ngrams_probs = {}
        for ngram in tqdm(ngrams):
            match self._smoothing_algorithm:

                case None:
                    cnt_ngram = self._ngrams_count.get(ngram, 0)
                    cnt_n1gram = self._ngrams_count.get(ngram[:-1], 1) if len(ngram) != 1 else self._num_tokens

                case 'laplace':
                    cnt_ngram = self._ngrams_count.get(ngram, 0) + 1
                    if len(ngram) != 1:
                        cnt_n1gram = self._ngrams_count.get(ngram[:-1], 0) + self._num_tokens_unique
                    else:
                        cnt_n1gram = self._num_tokens + self._num_tokens_unique

            ngrams_probs[ngram] = cnt_ngram / cnt_n1gram
        return ngrams_probs

    def train(self, tokens: list[str]) -> None:
        ngrams = self._get_ngrams(tokens)
        self._get_ngrams_count(ngrams)
        self.ngrams_probabilities = self._get_ngrams_probabilities(ngrams)

    @staticmethod
    def _count_perplexity(ngrams_probs: list, num_tokens_test: int) -> float:
        return 1 / 2 ** (sum(ngrams_probs) / num_tokens_test)

    def evaluate(self, tokens: list[str]):
        ngrams = self._get_ngrams(tokens)
        self._get_ngrams_probabilities(ngrams)
        ngrams_probs = self._get_ngrams_probabilities(ngrams)

        test_probs = []
        for ngram in ngrams:
            if len(ngram) != self.n:
                continue
            try:
                probability = log2(ngrams_probs[ngram])
            except ValueError:
                probability = 0
            test_probs.append(probability)

        return self._count_perplexity(test_probs, len(tokens))

    def generate_text(self,
                      first_word: str,
                      len_text: int,
                      generation_mode: str,
                      num_ngrams: int = None) -> str:

        curr_word = first_word
        text = [curr_word]
        match generation_mode:

            case 'most_probable':
                for _ in range(len_text):
                    ngrams_with_word = {ngram: prob for ngram, prob
                                        in self.ngrams_probabilities.items()
                                        if curr_word == ngram[0] and len(ngram) == self.n}

                    try:
                        most_probable_ngram = sorted(ngrams_with_word.items(),
                                                     key=lambda x: x[1],
                                                     reverse=True)[0][0]
                    except IndexError:
                        break

                    text.append(most_probable_ngram[-1])
                    curr_word = most_probable_ngram[-1]

            case 'random_next_word':
                for _ in range(len_text):
                    ngrams_with_word = {ngram: prob for ngram, prob
                                        in self.ngrams_probabilities.items()
                                        if curr_word == ngram[0] and len(ngram) == self.n}

                    error_flag = True
                    count_iter = 0
                    while error_flag:
                        try:
                            most_probable_ngram = sorted(ngrams_with_word.items(),
                                                         key=lambda x: x[1],
                                                         reverse=True)[random.randint(0, num_ngrams)][0]
                            error_flag = False
                        except IndexError:
                            continue
                        count_iter += 1
                        if count_iter > len_text:
                            break
                    if count_iter > len_text:
                        break

                    try:
                        text.append(most_probable_ngram[-1])
                        curr_word = most_probable_ngram[-1]
                    except NameError:
                        break

            case 'beam_search':
                ngrams_with_word = {ngram: prob for ngram, prob
                                    in self.ngrams_probabilities.items()
                                    if curr_word == ngram[0] and len(ngram) == self.n}
                most_probable_ngrams = sorted(ngrams_with_word.items(),
                                             key=lambda x: x[1],
                                             reverse=True)[:num_ngrams]
                sentences_dict = {}

                for ngram in most_probable_ngrams:
                    curr_word = ngram[0][-1]
                    curr_sentence = [curr_word]
                    curr_sentence_prob = 1.
                    curr_next_word = curr_word

                    for _ in range(len_text):
                        ngrams_with_word = {ngram: prob for ngram, prob
                                            in self.ngrams_probabilities.items()
                                            if curr_next_word == ngram[0] and len(ngram) == self.n}
                        most_probable_ngram = sorted(ngrams_with_word.items(),
                                                      key=lambda x: x[1],
                                                      reverse=True)[0]
                        curr_sentence.append(most_probable_ngram[0][-1])
                        curr_next_word = most_probable_ngram[0][-1]
                        curr_sentence_prob *= most_probable_ngram[1]

                    sentences_dict[tuple(curr_sentence)] = curr_sentence_prob

                most_probable_sentence = sorted(sentences_dict.items(),
                                                      key=lambda x: x[1],
                                                      reverse=True)[0][0]
                text.extend(most_probable_sentence)

            case 'beam_search_with_random':
                ngrams_with_word = {ngram: prob for ngram, prob
                                    in self.ngrams_probabilities.items()
                                    if curr_word == ngram[0] and len(ngram) == self.n}
                most_probable_ngrams = sorted(ngrams_with_word.items(),
                                              key=lambda x: x[1],
                                              reverse=True)[:random.randint(0, num_ngrams)]
                sentences_dict = {}

                for ngram in most_probable_ngrams:
                    curr_word = ngram[0][-1]
                    curr_sentence = [curr_word]
                    curr_sentence_prob = 1.
                    curr_next_word = curr_word

                    for _ in range(len_text):
                        ngrams_with_word = {ngram: prob for ngram, prob
                                            in self.ngrams_probabilities.items()
                                            if curr_next_word == ngram[0] and len(ngram) == self.n}
                        idx = random.randint(0, num_ngrams)
                        if idx > len(ngrams_with_word) - 1:
                            idx = random.randint(0, len(ngrams_with_word) - 1)
                        most_probable_ngram = sorted(ngrams_with_word.items(),
                                                     key=lambda x: x[1],
                                                     reverse=True)[idx]
                        curr_sentence.append(most_probable_ngram[0][-1])
                        curr_next_word = most_probable_ngram[0][-1]
                        curr_sentence_prob *= most_probable_ngram[1]

                    sentences_dict[tuple(curr_sentence)] = curr_sentence_prob

                most_probable_sentence = sorted(sentences_dict.items(),
                                                key=lambda x: x[1],
                                                reverse=True)[0][0]
                text.extend(most_probable_sentence)

        return ' '.join(text)
