# Bi-CSE

We proposed a bilingual sentence representation model which reached the state-of-the-art level in Chinese and English bilingual sentence representation tasks.

## Get Started

```
pip install -r requirements.txt
```

## Data prepare

download train file from  [google drive](https://drive.google.com/file/d/1pVdMAMd5VDQLPVuHeg-nyVrn6pxoEwEc/view?usp=sharing)
put it in data/train

## Train

```
bash train.sh
```

## Using HuggingFace Transformers

```
from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('zhou-xl/bi-cse')
model = AutoModel.from_pretrained('zhou-xl/bi-cse')
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)
```
