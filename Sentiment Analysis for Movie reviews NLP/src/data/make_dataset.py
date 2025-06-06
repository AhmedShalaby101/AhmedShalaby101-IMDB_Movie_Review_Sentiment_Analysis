import numpy as np
import pandas as pd
import os

df = {"train": [], "test": []}

data_dir = "../../data/raw/aclImdb"

## read dataset as dataframe

for split in ["train", "test"]:
    for sentiment in ["pos", "neg"]:
        path = os.path.join(data_dir, split, sentiment)
        label = 1 if sentiment == "pos" else 0
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                text_path = os.path.join(path, filename)
                with open(text_path, encoding="utf-8") as f:
                    review = f.read().strip()
                    df[split].append((review, label))

df_train = pd.DataFrame(df["train"])
df_test = pd.DataFrame(df["test"])
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
df.columns = ["review", "sentiment"]
df.head()

## we need to shuffle our data
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

## export dataset
df.to_csv("../../data/processed/preprocessed_data.csv", index=False)
