import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, AutoTokenizer, BertTokenizer
import numpy as np


class Model(nn.Module):
    """
    一个具有单隐藏层的多分类网络
    """
    def __init__(self, my_device):
        super(Model, self).__init__()

        self.student = AutoModel.from_pretrained('model_para/xlmr-base')
        self.tokenizer = AutoTokenizer.from_pretrained('model_para/xlmr-base')

        self.device = my_device

        self.dense = nn.Linear(1024, 1024)
        self.activation = nn.Tanh()
        self.mlp = False

    def forward(self, inputs):

        out_s = self.student(**inputs)

        if self.mlp:
            out_s = self.activation(self.dense(out_s))

        return out_s

    def encode(self, sentences, batch_size=64, max_length=512, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        list_tensor = list()
        num_sentence = len(sentences)
        if num_sentence % batch_size == 0:
            num_times = num_sentence//batch_size
        else:
            num_times = (num_sentence // batch_size) + 1

        for time in range(num_times):
            if time == num_times-1:
                this_sentences = sentences[time * batch_size:]
            else:
                this_sentences = sentences[time*batch_size:time*batch_size+batch_size]
            batch = self.tokenizer.batch_encode_plus(
                this_sentences,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=max_length,
            )

            # print(batch['input_ids'].shape)

            for k in batch:
                batch[k] = batch[k].to(self.device)
            with torch.no_grad():
                outputs = self.student(**batch, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.last_hidden_state
                embeddings = last_hidden[:, 0].cpu()  # bs x 1024
                embeddings = np.array(embeddings)
            for i in embeddings:
                list_tensor.append(i)
        return list_tensor
