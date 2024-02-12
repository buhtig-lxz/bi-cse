
from mteb.tasks import AmazonReviewsClassification, BUCCBitextMining, TatoebaBitextMining, STS12STS, STS13STS, STS14STS
from mteb.tasks import STS15STS, STS16STS, STSBenchmarkSTS, SickrSTS, STS22CrosslingualSTS, STS17Crosslingual, BiossesSTS
from C_MTEB import ATEC, BQ, LCQMC, PAWSX, STSB, QBQTC, AFQMC
import warnings
warnings.filterwarnings('ignore')
import datasets
from .evaluator import STSEvaluator

class MyBucc(BUCCBitextMining):
    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                'data/eval/bucc-bitext-mining',
                lang,
                revision=self.description.get("revision", None),
            )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        split = 'test'
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MyTatoeba(TatoebaBitextMining):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                'data/eval/tatoeba-bitext-mining',
                lang,
                revision=self.description.get("revision", None),
            )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        split = 'test'
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MySTS12(STS12STS):
    def __init__(self, **kwargs):
        super(MySTS12, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-en-mteb/sts12-sts', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MySTS13(STS13STS):
    def __init__(self, **kwargs):
        super(MySTS13, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-en-mteb/sts13-sts', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MySTS14(STS14STS):
    def __init__(self, **kwargs):
        super(MySTS14, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-en-mteb/sts14-sts', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MySTS15(STS15STS):
    def __init__(self, **kwargs):
        super(MySTS15, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-en-mteb/sts15-sts', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MySTS16(STS16STS):
    def __init__(self, **kwargs):
        super(MySTS16, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-en-mteb/sts16-sts', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MySTSB(STSBenchmarkSTS):
    def __init__(self, **kwargs):
        super(STSBenchmarkSTS, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-en-mteb/stsbenchmark-sts', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MySickR(SickrSTS):
    def __init__(self, **kwargs):
        super(SickrSTS, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-en-mteb/sickr-sts', revision=self.description.get("revision", None)
        )
        self.data_loaded = True
        # print(self.dataset)

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MyAFQMC(AFQMC):
    def __init__(self, **kwargs):
        super(AFQMC, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-zh-mteb/AFQMC/data', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    # @property
    # def description(self):
    #     return {
    #         "name": "AFQMC",
    #         "hf_hub_name": "C-MTEB/AFQMC",
    #         "type": "STS",
    #         "category": "s2s",
    #         "eval_splits": ["test"],
    #         "eval_langs": ["zh"],
    #         "main_score": "cosine_spearman",
    #         "min_score": 0,
    #         "max_score": 1,
    #     }

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MyATEC(ATEC):
    def __init__(self, **kwargs):
        super(ATEC, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-zh-mteb/ATEC/data', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

    def _evaluate_split(self, model, data_split, **kwargs):
        normalize = lambda x: (x - self.min_score) / (self.max_score - self.min_score)
        normalized_scores = list(map(normalize, data_split["score"]))
        evaluator = STSEvaluator(data_split["sentence1"], data_split["sentence2"], normalized_scores, limit=20000, **kwargs)
        metrics = evaluator(model)
        sent1, sent2, gold, pred = evaluator.sentences1, evaluator.sentences2, evaluator.gold_scores, evaluator.cosine_scores
        list_res = list(zip(sent1, sent2, gold, pred))
        list_res.sort(reverse=True, key=lambda x: abs(x[-1]-x[-2]))
        with open('result/ATEC.txt', 'w', encoding='utf-8') as f:
            for line in list_res:
                f.write(str(line)+'\n')
        return metrics

class MyBQ(BQ):
    def __init__(self, **kwargs):
        super(BQ, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-zh-mteb/BQ/data', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MyLCQMC(LCQMC):
    def __init__(self, **kwargs):
        super(LCQMC, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-zh-mteb/LCQMC/data', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MyPAWSX(PAWSX):
    def __init__(self, **kwargs):
        super(PAWSX, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-zh-mteb/PAWSX/data', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MyQBQTC(QBQTC):
    def __init__(self, **kwargs):
        super(QBQTC, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-zh-mteb/QBQTC/data', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MyZSTSB(STSB):
    def __init__(self, **kwargs):
        super(STSB, self).__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-zh-mteb/STSB/data', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

    def _evaluate_split(self, model, data_split, **kwargs):
        normalize = lambda x: (x - self.min_score) / (self.max_score - self.min_score)
        normalized_scores = list(map(normalize, data_split["score"]))
        evaluator = STSEvaluator(data_split["sentence1"], data_split["sentence2"], normalized_scores, limit=20000, **kwargs)
        metrics = evaluator(model)
        sent1, sent2, gold, pred = evaluator.sentences1, evaluator.sentences2, evaluator.gold_scores, evaluator.cosine_scores
        list_res = list(zip(sent1, sent2, gold, pred))
        list_res.sort(reverse=True, key=lambda x: abs(x[-1]-x[-2]))
        with open('result/STSBZ.txt', 'w', encoding='utf-8') as f:
            for line in list_res:
                f.write(str(line)+'\n')
        return metrics

class MySTS22(STS22CrosslingualSTS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                'data/eval/sts22-crosslingual-sts',
                lang,
                revision=self.description.get("revision", None),
            )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

    def _evaluate_split(self, model, data_split, **kwargs):
        normalize = lambda x: (x - self.min_score) / (self.max_score - self.min_score)
        normalized_scores = list(map(normalize, data_split["score"]))
        evaluator = STSEvaluator(data_split["sentence1"], data_split["sentence2"], normalized_scores, max_length=512, **kwargs)
        metrics = evaluator(model)
        return metrics

class MySTS17(STS17Crosslingual):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                'data/eval/sts17-crosslingual-sts',
                lang,
                revision=self.description.get("revision", None),
            )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

class MyBiossesSTS(BiossesSTS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            'data/eval/sts-en-mteb/biosses', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                # print(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            # print(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

    def _evaluate_split(self, model, data_split, **kwargs):
        normalize = lambda x: (x - self.min_score) / (self.max_score - self.min_score)
        normalized_scores = list(map(normalize, data_split["score"]))
        evaluator = STSEvaluator(data_split["sentence1"], data_split["sentence2"], normalized_scores, max_length=512, **kwargs)
        metrics = evaluator(model)
        return metrics


