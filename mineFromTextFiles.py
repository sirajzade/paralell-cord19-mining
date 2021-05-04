import sys
# !{sys.executable} -m spacy download en
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
import time

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


#warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


# Here are our functions

def sent_to_words(sentences):
  print ("preprocessing the sentences...")
  i = 1
  for sent in sentences:
    #sent = sent.decode('utf-8')
    #print (type(sent))
    sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
    sent = re.sub('\s+', ' ', sent)  # remove newline chars
    sent = re.sub("\'", "", sent)  # remove single quotes
    sent = re.sub("\u03b1", "", sent) # remove non unicode		
    sent = re.sub("\xb5","", sent) # remove non unicode for np.savetxt
    sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
    if i % 1000 == 0:
      print ("index for sentence preprocessing: " + str(i))
      #print (sent)
    i+=1
    yield(sent)

# !python3 -m spacy download en  # run in terminal once
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    #print ("This is for example the bigram data:")
    #print(bigram_mod)
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    #print ("this is the text after bigramis:")
    #print (str(texts).encode('utf-8'))
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'] )
    for sent in texts:
        #print (" type of sent: " + str(type(sent)) + "size of the text: " + str (len(sent)))
        stringSent = " ".join(sent)
        #print (" type of stringSent: " + str(type(stringSent)) + "size of the string: " + str (len(stringSent)))  
        if len(stringSent) > 1060900:
          print ("String is too big, cutting it into half: ")  
          stringSent = stringSent[:500000]
        doc = nlp(stringSent) 
        texts_out.append([token.lemma_ for token in doc]) # if token.pos_ in allowed_postags])
        #ents = [(e.text, e.label_, e.kb_id_) for e in doc.ents]
        #print(ents)
    # remove stopwords once more after lemmatization 
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

def format_topics_sentences(ldamodel=None, corpus=None, texts=None):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

import os, json
import pandas as pd

beginRead = time.time()
# this finds our json files
path_to_json = '/home/administrator/data/raw/Kaggle/document_parses/pdf_json/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# here I define my pandas Dataframe with the columns I want to get from the json
#jsons_data = pd.DataFrame(columns=['country', 'city', 'long/lat'])
real_text_data = []
# we need both the json and an index number so use enumerate()
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js), encoding="utf-8") as json_file:
        #print ("the name of the file: " + str(js))
        json_text = json.load(json_file)
        #print (type(json_text))
        ptext = json_text['body_text']
        #print(type(ptext))
        ctext = ""
        for mytext in ptext:
            ctext += mytext['text']
        #dtext = json.dumps(ctext, indent=4, sort_keys=False)
        #print("ctext is growing" + str(len(ctext)))
        real_text_data.append(str(ctext.encode('utf-8')))
        if index % 1000 == 0: 
            print ("index: " + str(index))
        if (index == 2000):
            break
endRead = time.time()
print("The files were opened and loaded into the memory in: " + str(endRead - beginRead) + " seconds")
print(type(real_text_data))
print(len(real_text_data))
  
 # print(titles[:10])

 
  # Convert to list
beginCleaningTime = time.time()
data_words = list(sent_to_words(real_text_data))
endCleaningTime = time.time()
  # print(data_words[:10])
print ("Data was cleaned in "+ str(endCleaningTime - beginCleaningTime) + " seconds") 
print("Preprocessing the data, building bigramms and trigramms...")
# Build the bigram and trigram models
beginBigramTime = time.time()
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
endBigramTime = time.time()
print ("bigrams are build in " + str(endBigramTime - beginBigramTime) + " seconds")
beginTrigramTime = time.time()
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
endTrigramTime = time.time()
print ("trigrams are build in " + str(endTrigramTime - beginTrigramTime) + " seconds")  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
print ("processing the words: ")
beginProcessTime = time.time()
data_ready = process_words(data_words)  # processed Text Data!
endProcessTime = time.time()
print ("processing the words took " + str(endProcessTime - beginProcessTime) + " seconds")


# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]
print("Start building an LDA model...")
beginLDATime = time.time()
# Build LDA model
lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=200, 
                                           random_state=100,
                                 #          update_every=1,
                                           chunksize=10,
					   workers=7,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

  #pprint(lda_model.print_topics())
endLDATime = time.time()
print("LDA done, it took " + str(endLDATime - beginLDATime) + " start writing files...")
with open("/home/administrator/data/python/topicsFromText.txt", "w") as log_file:
  pprint(lda_model.print_topics(-1, 10), log_file)

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

  # Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_selected = df_dominant_topic[['Document_No', 'Dominant_Topic']]
print (df_selected.head(100))
np.savetxt(r'/home/administrator/data/python/topToDocFromText.txt', df_selected.values, fmt='%s')







