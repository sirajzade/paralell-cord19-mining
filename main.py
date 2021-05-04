import readJSON as rs
import cleanTokenize as ct
import processWords as pw
import topicModeling as tm
import writeResults as wr
import time
import gensim 


<<<<<<< Updated upstream
path_to_json = 'document_parses/pdf_json/'
=======
>>>>>>> Stashed changes

# first we need to read all the json files
path_to_json = 'document_parses/pdf_json/'
files = rs.readJsonFiles(path_to_json, 1000)



# second we clean the files

cleaned_files = ct.clean_tokenize_multicore(files)


########### start building multi word units ############

#multi_worded_file = pw.multi_word_units_multicore(cleaned_files)

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
