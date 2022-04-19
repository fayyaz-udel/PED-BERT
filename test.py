import pandas as pd

from common.common import load_obj
from dataLoader.MLM import MLMLoader
from model.utils import age_vocab

code_df = pd.read_csv("./data/codes.csv", dtype=str).T.apply(lambda x: x.dropna().tolist()).tolist()
age_df = pd.read_csv("./data/ages.csv", dtype=str).T.apply(lambda x: x.dropna().tolist()).tolist()

ageVocab, _ = age_vocab(max_age=100, mon=1, symbol=None)

BertVocab = load_obj("./data/dict")

Dset = MLMLoader({"code": code_df, "age": age_df}, BertVocab['token2idx'], ageVocab, max_len=300)

for i in range(20):
    print(Dset[i])
print("End")