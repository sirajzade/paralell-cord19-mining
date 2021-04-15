import time
import re
import gensim

# This function cleans the text data; it replaces not usefull information like emails
# also it cleans undesired characters 

def clean_tokenize(documents: list)->list:
  retval = []
  i = 0
  for doc in documents:
    #print(type(doc))
    #doc = doc.decode('utf-8')
    #print (type(doc))
    doc = re.sub('\S*@\S*\s?', '', doc)  # remove emails
    doc = re.sub('\s+', ' ', doc)  # remove newline chars
    doc = re.sub("\'", "", doc)  # remove single quotes
    doc = re.sub("\u03b1", "", doc) # remove non unicode   
    doc = re.sub("\xb5","", doc) # remove non unicode for np.savetxt
    doc = gensim.utils.simple_preprocess(str(doc), deacc=True)
    #if i % 10 == 0:
      #print ("index for document preprocessing: " + str(i))
      #print (doc)
    #i+=1
    #print (i)
    retval.append(doc)
  return retval




def clean_tokenize_multicore(doc: str)->str:
  i = 0
    #doc = doc.decode('utf-8')
    #print (type(doc))
  doc = re.sub('\S*@\S*\s?', '', doc)  # remove emails
  doc = re.sub('\s+', ' ', doc)  # remove newline chars
  doc = re.sub("\'", "", doc)  # remove single quotes
  doc = re.sub("\u03b1", "", doc) # remove non unicode   
  doc = re.sub("\xb5","", doc) # remove non unicode for np.savetxt
  doc = gensim.utils.simple_preprocess(str(doc), deacc=True)
  #if i % 10 == 0:
    #print ("index for document preprocessing: " + str(i))
      #print (doc)
  #i+=1
  #print (i)
  return doc