import topicModeling as tm
import writeResults as wr

import time
import gensim 


id2word = gensim.corpora.Dictionary.load('id2word.dict')
mm = gensim.corpora.MmCorpus('myBigCorpus.mm')

lda_model = tm.topic_modeling(cleaned_files, id2word, mm)

df_topic_doc_keywords = wr.format_topics_documents(lda_model, corpus, cleaned_files)

topicsFile = "/home/administrator/data/python/topicsFromText.txt"

top2docFile = '/home/administrator/data/python/topToDocFromText.txt'

wr.write_results (lda_model, df_topic_doc_keywords, topicsFile, top2docFile)

print ("the script ended at ", time.ctime(time.time()))
