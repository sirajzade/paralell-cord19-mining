import time
import re
import gensim
from multiprocessing import Pool

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




def clean_tokenize_function(doc: str)->str:
  i = 0
  #print (multiprocessing.current_process())
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


"""
This method is for multicore cleaning

"""
def clean_tokenize_multicore(files:list)->list:
  ############# start to reduce cleaning time ###################

  beginCleaningTimeMultiCore = time.time()
  print ("Multicore cleaning the documents...startet at " + str(time.ctime(time.time())))
  with Pool(4) as p:
    cleaned_files = p.map(clean_tokenize_function, files)
    print("Pool was startet: ")
  endCleaningTimeMultiCore = time.time()
  print ("Data was multicore cleaned in "+ str(endCleaningTimeMultiCore - beginCleaningTimeMultiCore) + " seconds")
  #print (str(len(cleaned_files)))
  #print (result[0])
  return cleaned_files



