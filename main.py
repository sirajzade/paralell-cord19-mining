import readJSON as rs
import cleanTokenize as ct
import processWords as pw
import topicModeling as tm
import writeResults as wr
import time
import gensim 
from multiprocessing import Pool


path_to_json = '/home/administrator/data/raw/Kaggle/document_parses/pdf_json/'

files = rs.readJsonFiles(path_to_json, 1000)



############# start to reduce cleaning time ###################

beginCleaningTimeMultiCore = time.time()
print ("Multicore cleaning the documents...")
with Pool(6) as p:
    cleaned_files = p.map(ct.clean_tokenize_multicore, files)
endCleaningTimeMultiCore = time.time()
print ("Data was multicore cleaned in "+ str(endCleaningTimeMultiCore - beginCleaningTimeMultiCore) + " seconds")
print (str(len(cleaned_files)))
#print (result[0])
 

#beginCleaningTime = time.time()
#print ("One core cleaning the documents...")
#cleaned_files = ct.clean_tokenize(files)
#endCleaningTime = time.time()
#print ("Data was one core cleaned in "+ str(endCleaningTime - beginCleaningTime) + " seconds")
#print (str(len(cleaned_files)))
#print (cleaned_files[0])


########### start to reduce building multi word units ############

multi_worded_file = pw.multi_word_units_multicore(cleaned_files)


#multi_worded_file = pw.multi_word_units(cleaned_files)





#lemmatized_file = pw.tag_lemmatize(multi_worded_file)

# Create Dictionary
#id2word = gensim.corpora.Dictionary(lemmatized_file)

# Create Corpus: Term Document Frequency
#corpus = [id2word.doc2bow(doc) for doc in lemmatized_file]

#lda_model = tm.topic_modeling(lemmatized_file, id2word, corpus)

#df_topic_doc_keywords = wr.format_topics_documents(lda_model, corpus, lemmatized_file)

topicsFile = "/home/administrator/data/python/topicsFromText.txt"

top2docFile = '/home/administrator/data/python/topToDocFromText.txt'

#wr.write_results (lda_model, df_topic_doc_keywords, topicsFile, top2docFile)