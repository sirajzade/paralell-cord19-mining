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

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


# Hier kommen die Funktionen

def sent_to_words(sentences):
  for sent in sentences:
    sent = sent.decode('utf-8')
    sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
    sent = re.sub('\s+', ' ', sent)  # remove newline chars
    sent = re.sub("\'", "", sent)  # remove single quotes
    sent = re.sub("\u03b1", "", sent) # remove non unicode		
    sent = re.sub("\xb5","", sent) # remove non unicode for np.savetxt
    sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
    yield(sent)

# !python3 -m spacy download en  # run in terminal once
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
 #   print(bigram_mod)
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'] )
    for sent in texts:
        doc = nlp(" ".join(sent)) 
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




import mysql.connector
from mysql.connector import errorcode

try:
    cnx = mysql.connector.connect(user='deephouse', database='metadata', password='deephouse2020', host='127.0.0.1')

except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
else:
  cursor = cnx.cursor()

  query = ("SELECT abstract FROM metadata " 
        ";")#"LIMIT 1000;")

  query2 = ("SELECT abstract " 
	"FROM metadata.metadata as m1 "  
	"INNER JOIN "
	"(SELECT journal, count(cord_uid) as anzahl " 
    	"FROM metadata.metadata "
    	"WHERE journal != '' and abstract !='' "
    	"GROUP BY journal "
    	"ORDER BY anzahl desc limit 10) as m2 "
	"ON m1.journal = m2.journal "
	"WHERE m1.abstract !='' "
	"ORDER BY m1.journal LIMIT 30000;"
	)

  cursor.execute(query2)

  titles = []

  for (title) in cursor:
#    print(''.join(title).encode('utf-8'))
    titles.append(''.join(title).encode('utf-8'))
  
 # print(titles[:10])

 
  # Convert to list
  data_words = list(sent_to_words(titles))
  # print(data_words[:10])
 
  print("Preprocessing the data, building bigramms and trigramms...")
# Build the bigram and trigram models
  bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
  trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
  bigram_mod = gensim.models.phrases.Phraser(bigram)
  trigram_mod = gensim.models.phrases.Phraser(trigram)

  data_ready = process_words(data_words)  # processed Text Data!

  #print(data_ready[:10])
  cursor.close() 
  cnx.close()



# Create Dictionary
  id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
  corpus = [id2word.doc2bow(text) for text in data_ready]
  print("Start building an LDA model...")
# Build LDA model
  lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=40, 
                                           random_state=100,
                                 #          update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

  #pprint(lda_model.print_topics())
  print("LDA done, start writing files...")
  with open("/home/administrator/data/python/topicsAllAbstr.txt", "w") as log_file:
    pprint(lda_model.print_topics(-1, 10), log_file)

  df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

  # Format
  df_dominant_topic = df_topic_sents_keywords.reset_index()
  df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
  df_selected = df_dominant_topic[['Document_No', 'Dominant_Topic']]
  print (df_selected.head(100))
  np.savetxt(r'/home/administrator/data/python/topToDocAllAbstr.txt', df_selected.values, fmt='%s')








