import time 
import gensim
import spacy 
# NLTK Stop words
from nltk.corpus import stopwords
from multiprocessing import Pool, Process, Value, Array

# here we put additional information to each word
# we also try to find phraseological units of bigrams and trigrams

# In goes a list of documents
# Out comes a list of the same documents but enriched with multi word units 

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


def multi_word_units (documents:list) -> list:
   #print (documents[0])
   print("Preprocessing the data, building multiword units of bigramms and trigramms...")
   # Build the bigram and trigram models
   beginBigramTime = time.time()
   bigram = gensim.models.Phrases(documents, min_count=5, threshold=100) # higher threshold fewer phrases.
   endBigramTime = time.time()
   print ("bigrams are build in " + str(endBigramTime - beginBigramTime) + " seconds")
   beginTrigramTime = time.time()
   trigram = gensim.models.Phrases(bigram[documents], threshold=100)
   endTrigramTime = time.time()
   print ("trigrams are build in " + str(endTrigramTime - beginTrigramTime) + " seconds")  
   bigram_mod = gensim.models.phrases.Phraser(bigram)
   trigram_mod = gensim.models.phrases.Phraser(trigram)
   print ("adding multi word units to the documents: ")
   beginProcessTime = time.time()
   
   # put the data in faster Frozen Phraser in order save resources

   bigram_mod = gensim.models.phrases.Phraser(bigram)
   trigram_mod = gensim.models.phrases.Phraser(trigram)

   documents = [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in documents]
   documents = [bigram_mod[doc] for doc in documents]
   documents = [trigram_mod[bigram_mod[doc]] for doc in documents]

   #data_ready = process_words(documents)  # processed Text Data!
   #print (documents[0])
   endProcessTime = time.time()
   print ("Building multiword Units of the words took " + str(endProcessTime - beginProcessTime) + " seconds")
   return documents


# here we lemmatize the text, it is similar to stemming
# In goes a list of documents
# Out comes a list of the same documents but selected with part of speech and lemmatized

def tag_lemmatize (documents:list) -> list:
   begin_lemma_time = time.time()
   print ("lemmatizing the text: ")   
   documents_out = []
   nlp = spacy.load('en', disable=['parser', 'ner'] )
   for document in documents:
        #print (" type of sent: " + str(type(sent)) + "size of the text: " + str (len(sent)))
      stringDocument = " ".join(document)
        #print (" type of tsringSent: " + str(type(stringSent)) + "size of the string: " + str (len(stringSent)))  
      if len(stringDocument) > 1060900:
         print ("String is too big, cutting documnets into half: ")  
         stringDocument = stringDocument[:500000]
      doc = nlp(stringDocument) 
      documents_out.append([token.lemma_ for token in doc]) # if token.pos_ in allowed_postags])
      #ents = [(e.text, e.label_, e.kb_id_) for e in doc.ents]
      #print(ents)
    # remove stopwords once more after lemmatization 
   documents_out = [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in documents_out]    
   #print (documents_out[0])
   end_lemma_time = time.time()
   print ("lemmatizing took " + str(end_lemma_time - begin_lemma_time) + " seconds ")
   return documents_out


def makePhrasesCallabe(phrases, documents):
   phrases = gensim.models.Phrases(documents, min_count=5, threshold=100) 


def multi_word_units_multicore (documents:list) -> list:
   #print (documents[0])
   print("Preprocessing the data, building multiword units of bigramms and trigramms...")
   # Build the bigram and trigram models
   beginBigramTimeMulticore = time.time()
   #devided = chunkIt(documents, 6)
   
   num = Array('i', gensim.models.Phrases)
   arr = Array('i', documents)
   p = Process(target=makePhrasesCallabe, args=(num, arr))
   p.start()
   p.join()

   endBigramTimeMulticore = time.time()
   #print ("Building multiword Units of the words took " + str(endProcessTime - beginProcessTime) + " seconds")
   return documents


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


