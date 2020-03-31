# import pickle and write necessary pickle transformation functions

import pickle
def writePickle( Variable, fname):
    filename = fname +".pkl"
    f = open("pickle_vars/"+filename, 'wb')
    pickle.dump(Variable, f)
    f.close()
def readPickle(fname):
    filename = "pickle_vars/"+fname +".pkl" # notice the ../ addition to reach out to variables from the parent directory
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

# import beautifulsoup
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2 
from bs4 import BeautifulSoup

# import requests and time
import requests
import time


# read the ids to lyrics dict to get song lyrics to be converted to phonemes
ids_to_lyrics = readPickle("final_IDs_to_Lyrics_dict")
length = len(list(ids_to_lyrics.keys()))

print("There are a total of", length, "songs to be converted to phonemes")

# main script that accesses a web page and converts lyrics to phonemes

id2phoneme_dict = dict()
url = 'http://upodn.com/phon.php' # the web site used for phoneme translation. all translations in this script are made in British English
problematic_ids = list()
counter = 0
for ids, lyrics in ids_to_lyrics.items():
    counter += 1
    print(counter, "out of", length, "songs is being processed. The song id being processed now is:", ids)
    lines = lyrics.split('<>')
    text = ""
    # check whether the second line gives song writer info. if so, remove the first two lines
    if lines[1][0:6] == 'Writer' or lines[1][0:7] == 'Writers':
        lines = lines[2:]
    for line in lines[:-1]:
        text += line
        text += " ohohohohoh " # to mark a unique separator
    post_params = {'intext': text}
    response = requests.post(url, data=post_params)
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        phoneme_translation = soup.findAll('td', {'align': "left"})[0].get_text()
        id2phoneme_dict[ids] = phoneme_translation
    except:
        problematic_ids.append(ids)        
    
    if counter%150==0:
        writePickle(id2phoneme_dict, "final_IDs_to_Phonemes_dict")
        print(phoneme_translation)
        time.sleep(15)
        


print("The last phase of translation: Writing the phoneme dictionaries to pickle files...")
writePickle(id2phoneme_dict, "final_IDs_to_Phonemes_dict")
    
