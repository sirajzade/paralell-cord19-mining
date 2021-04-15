from pprint import pprint
import gensim.models.ldamulticore as lda
import numpy as np 
import pandas as pd

def write_results(lda_model:lda.LdaMulticore, df_topic_doc_keywords, topicsFile:str, topicToDocFile:str): 
   # Format
  df_dominant_topic = df_topic_doc_keywords.reset_index()
  df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
  df_selected = df_dominant_topic[['Document_No', 'Dominant_Topic']]
  #print (df_selected.head(100))
  np.savetxt(topicToDocFile, df_selected.values, fmt='%s')

  with open(topicsFile, "w") as file:
    pprint(lda_model.print_topics(-1, 10), file)



def format_topics_documents(ldamodel, corpus, documents):

    # Init output
    doc_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
      row = row_list[0] if ldamodel.per_word_topics else row_list            
      #print(row)
      row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
      for j, (topic_num, prop_topic) in enumerate(row):
        if j == 0:  # => dominant topic
          wp = ldamodel.show_topic(topic_num)
          topic_keywords = ", ".join([word for word, prop in wp])
          doc_topics_df = doc_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
        else:
          break
    doc_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(documents)
    doc_topics_df = pd.concat([doc_topics_df, contents], axis=1)
    return(doc_topics_df)