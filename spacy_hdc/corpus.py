import spacy
from tqdm import tqdm
import logging
tqdm.pandas()
import pandas as pd
import numpy as np
import sys, os, re, json, glob
import textwrap

from .helpers.filter import clean_text
from spacy.language import Language as SpacyLang
tqdm.monitor_interval=0

from textacy import Corpus

class CorpusHDC:

    def __init__(self,text_df,name,lang,chunk_size=500):
        self.text_df = text_df
        self.text_df["content"]=self.text_df.content.progress_apply(lambda x: clean_text(x))
        self.name = name
        self.chunk_size = chunk_size

        self.text_split_size=10000 # faster 

        if isinstance(lang,str):
            self.nlp=spacy.load(lang)
        if isinstance(lang,SpacyLang):
            self.nlp=lang

    def fit_text(self):
        new_data=[]
        for idx,row in tqdm(self.text_df.iterrows(),total=len(self.text_df)):
            if len(row.content) <10000:
                new_data.append([idx,row.id_doc,row.content])
            else:
                texts=textwrap.wrap(row.content,10000)
                for t in texts:
                    new_data.append([idx,row.id_doc,t])
        return new_data

    def create_corpus(self):
        new_format=self.fit_text()
        self.text_df=pd.DataFrame(new_format,columns="origin_id id_doc content".split())
        chunks=np.array_split(self.text_df.content.values.tolist(), round(len(self.text_df)/self.chunk_size))
        chunks_ids=np.array_split(list(range(len(self.text_df))),round(len(self.text_df)/self.chunk_size))
        chunks_ids=np.array([[i]*len(ch) for i,ch in enumerate(chunks_ids)]).flatten()
        flat_=[]
        for c in chunks_ids:
            flat_.extend(c)
        chunks_ids = flat_
        self.text_df["chunks_id"]=chunks_ids

        prev_chunk=0
        for chunk in tqdm(chunks):
            next_chunk_size=prev_chunk+len(chunk)
            fn="{2}_{0}to{1}.pkl".format(prev_chunk,next_chunk_size,self.name)
            cor_en=Corpus(self.nlp,chunk.tolist())
            cor_en.save(fn)
            prev_chunk=next_chunk_size
    
    def save(self):
        pass
    @staticmethod
    def load(self):
        pass


