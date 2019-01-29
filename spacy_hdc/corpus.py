import spacy
from tqdm import tqdm
import logging
tqdm.pandas()
import pandas as pd
import numpy as np
import sys, os, re, json, glob
import textwrap
import pickle

from .helpers.filter import clean_text
from spacy.language import Language as SpacyLang
tqdm.monitor_interval=0

from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP
from spacy.tokens import Doc

class Corpus():

    def __init__(self,name,text_list,lang):
        self.name=name
        self.nlp=lang
        self.text_list=text_list
        self.doc_bytes=[]
        self.text_bytes=[]
        self.vocab_bytes=None
        self.noun_chunks=[]

    def extract_data(self,attr=[LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP]):
        for doc in self.nlp.pipe(self.text_list,batch_size=100, n_threads=4):
            self.doc_bytes.append(doc.to_array(attr))
            self.text_bytes.append([t.text for t in doc])
            self.noun_chunks.append([[nc.text,nc.start,nc.end] for nc in doc.noun_chunks])
        self.vocab_bytes=self.nlp.vocab.to_bytes()

    def save(self):
        with open("{0}.pickle".format(self.name),"wb") as handle:
            pickle.dump((self.doc_bytes, self.vocab_bytes,self.text_bytes,self.noun_chunks),handle)
            #doc_bytes = [doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA,DEP]) for doc in docs]
    @staticmethod
    def load(fn,nlp):
        with open(fn, "rb") as handle:
            doc_bytes, vocab_bytes,text_bytes,noun_chunks = pickle.load(handle)
            nlp.vocab.from_bytes(vocab_bytes)
            docs = [Doc(nlp.vocab,words=text_bytes[i]).from_array([LOWER, POS, ENT_TYPE, IS_ALPHA,DEP],b) for i,b in enumerate(doc_bytes)]
        return docs,noun_chunks
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
            fn="{2}_{0}to{1}".format(prev_chunk,next_chunk_size,self.name)
            cor_en=Corpus(fn,chunk.tolist(),self.nlp)
            cor_en.extract_data()
            cor_en.save()
            prev_chunk=next_chunk_size
    
    def save(self):
        self.text_df.to_csv("{0}_{1}.csv".format(self.name,self.chunk_size))


