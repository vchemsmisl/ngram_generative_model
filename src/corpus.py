
class Coprus:

    def __init__(self, corpus_path: str) -> None:
        self.corpus = self.open_corpus(corpus_path)

    @staticmethod
    def open_corpus(corpus_path) -> str:
        with open(corpus_path, 'r', encoding='utf-8') as file:
            return file.read()

    def split_corpus(self, train_size=0.6) -> tuple[str, str]:
        split_thresh = round(len(self.corpus) * train_size)
        return self.corpus[:split_thresh], self.corpus[split_thresh:]
