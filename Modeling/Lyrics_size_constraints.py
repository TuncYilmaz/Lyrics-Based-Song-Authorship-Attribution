# start by feeding your Pickle functions to call and save pickle variables to be used later on

import pickle
def writePickle( Variable, fname):
    filename = fname +".pkl"
    f = open("pickle_vars/"+filename, 'wb')
    pickle.dump(Variable, f)
    f.close()
def readPickle(fname):
    filename = "pickle_vars/"+fname +".pkl"
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj
def readPicklefromPast(fname):
    filename = "../pickle_vars/"+fname +".pkl"
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

# load the previously formed pickle variable that maps each song id to its lyrics
ids_to_lyrics_dict = readPicklefromPast("all_songids2_lyrics")
# also load the previously formed pickle variable that contains all artists with more than 150 english songs
artists_and_english_song_ids = readPickle("artists_and_english_song_ids")

# import the spacy english language model
import en_core_web_sm
nlp = en_core_web_sm.load()

counter = 0
size_constrained_artists_to_ids = dict()
for artist, song_list in artists_and_english_song_ids.items():
    size_constrained_artists_to_ids[artist] = list()
    for song_id in song_list:
        counter +=1
        print(counter)
        lines = ids_to_lyrics_dict[song_id].split('<>')
        # check whether the second line gives song writer info. if so, remove the first two lines
        if lines[1][0:6] == 'Writer' or lines[1][0:7] == 'Writers':
            lines = lines[2:]
        song_length = len(lines)-1
        # if the song_length is below 10 or above 100, ignore song and continue with the next one
        if song_length < 10 or song_length > 100:
            continue
        else: # if the song length satisfies our constraints, continue checking the max line length
            line_length_counter = []
            for line in lines[:-1]: # because the last line is always blank in the dataset
                doc = nlp(line)
                tokens = []
                for token in doc:
                    tokens.append(token.text)
                line_length_counter.append(len(tokens))
        if max(line_length_counter) < 21: # as long as the max line length in a song is less than 21, add the song id
            size_constrained_artists_to_ids[artist].append(song_id)

writePickle(size_constrained_artists_to_ids, "size_constrained_artists_to_ids")
