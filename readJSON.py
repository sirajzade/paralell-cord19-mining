import os
import time
import json

# this finds our json files

def readJsonFiles (json_dir:str, max_number_files:int) -> list: 
    beginRead = time.time()
    json_files = [pos_json for pos_json in os.listdir(json_dir) if pos_json.endswith('.json')]

    real_text_data = []
    # we need both the json and an index number so use enumerate()
    for index, js in enumerate(json_files):
        with open(os.path.join(json_dir, js), encoding="utf-8") as json_file:
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
            if (index == max_number_files):
                break
    endRead = time.time()
    print("The files were opened and loaded into the memory in: " + str(endRead - beginRead) + " seconds")
    #print("The type of the output: " + str(type(real_text_data)))
    #print("The size of the output: " + str(len(real_text_data)))
    return real_text_data
