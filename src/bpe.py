from collections import Counter


class BPETokenizer:
    def __init__(self, corpus, merge_count):
        self.corpus = corpus
        self.merge_count = merge_count
        self.word_freq = Counter()
        self.splits = {}
        self.merges = {}
        self.vocab = set()
        self.init_splits_vocab_word_freq()
        self.merge()

    def init_splits_vocab_word_freq(self):
        for text in self.corpus:
            words = text.split()
            words = [word + "_" for word in words]
            self.word_freq.update(words)
            for word in words:
                split = []
                for c in word:
                    split.append(c)
                    self.vocab.add(c)
                self.splits[word] = split

    def max_pair(self):
        pair_freq = Counter()
        for word in self.splits:
            split = self.splits[word]
            for i in range(len(split) - 1):
                pair_freq[(split[i], split[i+1])] += self.word_freq[word]
        return pair_freq.most_common()[0][0]

    @staticmethod
    def merge_pair_with_split(split, a, b):
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        return split

    def merge_pair(self, a, b):
        for word in self.splits:
            split = self.splits[word]
            self.splits[word] = self.merge_pair_with_split(split, a, b)

    def merge(self):
        for i in range(self.merge_count):
            try:
                a, b = self.max_pair()
            except IndexError:
                print(f"Maximum merge Count is {i}! Stop Merging!")
                break
            self.merge_pair(a, b)
            self.merges[(a, b)] = a+b
            self.vocab.add(a+b)

    def tokenize(self, text):
        words = text.split()
        words = [word+"_" for word in words]
        splits = [[c for c in word] for word in words]
        for pair in self.merges:
            a, b = pair
            for idx, split in enumerate(splits):
                splits[idx] = self.merge_pair_with_split(split, a, b)
        return sum(splits, [])


if __name__ == '__main__':
    corpus = [
        "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
    ]
    merge_count = 17
    text_sentence = "newer lower"
    tokenizer = BPETokenizer(corpus, merge_count)
    print(tokenizer.vocab)
    print(tokenizer.merges)
    print(tokenizer.tokenize(text_sentence))
