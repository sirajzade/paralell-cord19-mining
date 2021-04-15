import gensim
import time

# here we build frequencies from the corpus
# In goes a list of documents
# Out comes a trained gensim model

def topic_modeling (documents:list, id2word, corpus):

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
  return lda_model