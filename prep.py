import readJSON as rs
import cleanTokenize as ct
import processWords as pw

import time
import gensim 

print ("the script was started at ", time.ctime(time.time()))
# first we need to read all the json files
path_to_json = 'document_parses/pdf_json/'
files = rs.readJsonFiles(path_to_json, 10000)

# second we clean the files

cleaned_files = ct.clean_tokenize_multicore(files)

########### start building multi word units ############

#multi_worded_file = pw.multi_word_units_bi(cleaned_files)

#multi_worded_file = pw.multi_word_units(cleaned_files)

#lemmatized_file = pw.tag_lemmatize(multi_worded_file)

# Create Dictionary
beginId2wordTime = time.time()
id2word = gensim.corpora.Dictionary(cleaned_files)
endId2wordTime = time.time()
id2word.save('id2word.dict');
print("id2word took " + str(endId2wordTime - beginId2wordTime))
# Create Corpus: Term Document Frequency
beginCreateCorpus = time.time()
corpus = [id2word.doc2bow(doc) for doc in cleaned_files]
gensim.corpora.MmCorpus.serialize('myBigCorpus.mm', corpus)
endCreateCorpus = time.time()
print(" Creating a corpus took  " + str(endCreateCorpus - beginCreateCorpus))