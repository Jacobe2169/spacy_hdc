import spacy
from tqdm import tqdm
import logging
tqdm.pandas()
import pandas as pd
import numpy as np
import sys, os, re, json, glob
import textwrap

df=pd.read_csv("/Users/jacquesfize/nas_cloud/Code/thematic_str/data/corpus/bv_lac_corpus.csv",index_col=0)
df["lang"]=df.lang.apply(lambda x: "en" if x not in ["en","fr"] else x)

df_en=df[df.lang == "en"]
df_fr=df[df.lang == "fr"]

import spacy
nlp_models = {lang:spacy.load(lang,disable=["ner","textcat"]) for lang in pd.unique(df.lang)}


from spacy_hdc.corpus import CorpusHDC

corpus_en=CorpusHDC(df_en,"en_corpus/doc",nlp_models["en"])
corpus_en.create_corpus()
corpus_en.save()
