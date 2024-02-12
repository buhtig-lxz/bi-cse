
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, AutoTokenizer

class CustomDataset(Dataset):
    """
    用于准备训练数据
    """
    def __init__(self, list_sample):
        self.list_sample = list_sample

    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, index):
        return self.list_sample[index]

class Collate_fn():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        list_sentence = list()
        for line in batch:
            list_sentence.append(line[0])
        for line in batch:
            list_sentence.append(line[1])
        for line in batch:
            list_sentence.append(line[2])

        inputs = self.tokenizer.batch_encode_plus(list_sentence, padding=True, max_length=64, return_tensors='pt',
                                                  truncation=True)
        return inputs, torch.tensor([0])

    # 二元组
    # def __call__(self, batch):
    #
    #     list_sentence = list()
    #     list_label = list()
    #     for line in batch:
    #         list_sentence.append(line[0])
    #         list_label.append(line[-1])
    #     for line in batch:
    #         list_sentence.append(line[1])
    #
    #     inputs = self.tokenizer.batch_encode_plus(list_sentence, padding=True, max_length=64, return_tensors='pt',
    #                                               truncation=True)
    #     label = torch.tensor(list_label)
    #     return inputs, label
