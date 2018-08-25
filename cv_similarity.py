from scipy import spatial
import gensim
from gensim.models import Word2Vec
from rake_nltk import Rake
import sklearn
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
import nltk 
import numpy as np
import pandas as pd
import textract
import string
import collections
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
keyword_extractor=Rake(stopwords=stop_words,punctuations=string.punctuation)
lemmatizer = WordNetLemmatizer()
feature_vector=sklearn.feature_extraction.text.TfidfVectorizer(min_df=0,max_df=50)



def file_processing(data):
    '''
    used for processing file
    Parameters:
    The text file path
    '''
    text=textract.process(data)
    text=text.decode("utf-8")
    text=text.lower()
    return text

def keyword_extractor_text(data):
    '''
    This Function is used for Extracting Keywords
    
    Parameters:
    The tokenised form of the text
    '''
    keyword_extractor.extract_keywords_from_text(data)
    keywords=keyword_extractor.ranked_phrases
    keywords_tokens=''.join(c+" " for c in keywords)
    keywords_tokens=nltk.word_tokenize(keywords_tokens)
    text_tokens=nltk.word_tokenize(data)
    text_tokens = [s for s in text_tokens if s]
    final_text=[w for w in text_tokens if w in keywords_tokens]
    final_text =[c for c in final_text if c not in string.punctuation]
    final_text=[w for w in final_text if w.isalpha()]
    final_text=[lemmatizer.lemmatize(s) for s in final_text]
    return final_text




def TFIDF(job_req,cv):
    '''
    This function is used to calculate TFIDF vector
    Parameters:
    Job requirements keywords,cv keywords
    '''
    job_lemmatization=[lemmatizer.lemmatize(s) for s in job_req]
    cv_lemmatization=[lemmatizer.lemmatize(s) for s in cv]
    job_req_feature=feature_vector.fit_transform(job_lemmatization).toarray()
    cv_feature=feature_vector.transform(cv_lemmatization).toarray()
    return job_req_feature,cv_feature



def jaccard_similarity(job_req,cv):
    '''
    CAN BE IMPLEMENTED ONLY WITH TFIDF OR SPARSE MATRIX!!!!!!
    This function is used to calculate jaccard_similarity
    Parameters:
    Job req TFIDF matrix,cv TFIDF matrix
    '''
    cnt=0
    for j in range(cv.shape[1]):
        for i in range(cv.shape[0]):
            if cv[i][j]==1:
                cnt=cnt+1
                break
    similarity=(cnt/job_req.shape[1])*100
    return similarity



def cosine_similarity_value(job_req,cv,matrix):
    '''
    This function calculates cosine similarity and it works with both TFIDF vector and word embeddings but not a good measure 
    Parameters:
    Job req:TFIDF or embeddings matrix,cv:TFIDF or embeddings matrix,type of matrix:embeddings or TFIDF
    '''
    if matrix=="TFIDF":
        similarity=cosine_similarity(job_req,cv)
        similarity=similarity.mean()*100
    elif matrix=="embeddings":
        similarity=spatial.distance.cosine(job_req,cv)*100
    return similarity


def word_embeddings(job_req,cv):
    '''
    only used for calculating spatial cosine distance 
    '''
    vector_job=[]
    vector_cv=[]
    model=Word2Vec.load("word2vec_model")
    for i in job_req:
        if i in model.wv.vocab:
            vector_job.append(model.wv[i])
    for i in cv:
        if i in model.wv.vocab:
            vector_cv.append(model.wv[i])
    return  np.mean(vector_job,axis=0),np.mean(vector_cv,axis=0)



file1=file_processing("/home/rahul/machine learning/Freeflo /text-analyser/cv.txt")
keywords1=keyword_extractor_text(file1)
file2=file_processing("/home/rahul/machine learning/Freeflo /text-analyser/job_req.txt")
keywords2=keyword_extractor_text(file2)
vector1,vector2=TFIDF(keywords2,keywords1)


print("cosine similarity:")
print(cosine_similarity_value(vector1,vector2,"TFIDF"))
print("jaccard similarity:")
print(jaccard_similarity(vector1,vector2))

