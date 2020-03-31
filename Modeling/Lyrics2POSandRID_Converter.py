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

# import necessary spacy packages and models
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

# load necessary pickle variables
final_artist2lyrics_dict =  readPickle("final_artist2lyrics_dict") # this is the dictionary mapping each artist to a lyrics list consisting of 150 song lyrics
final_constrained_artist2idlist_dict = readPickle("final_constrained_artist2idlist_dict") # this is the dictionary mapping each artist to a lyrics list consisting of 150 song ids
RID = readPickle("RID") # this is the RID dictionary
print("Variables are read")


# define song size constraints
max_song = 100
max_line = 20

# for RID conversion, create other dictionaries and lists that will come handy

# this script mapped all the adjectives in the RID dictionary to their main categories
reverse_RID = dict()
for category, adjective_list in RID.items():
    for adjective in adjective_list:
        reverse_RID[adjective.lower()] = category # don't forget lowercase

# the following script further processes the adjectives, putting them into two distinct categories
RID_strict_adjectives = list()
RID_flex_adjectives = list()
for adjective in list(reverse_RID.keys()):
    if adjective[-1] == "*":
        RID_flex_adjectives.append(adjective[:-1])
    else:
        RID_strict_adjectives.append(adjective)

print("RID adjective conversion is completed")


# now continue with the RID and POS conversion
counter = 0 # use a counter to report progress


final_artists_to_RIDsongs_dict = dict()
final_artists_to_POSsongs_dict = dict()
for artist, lyrics_list in final_artist2lyrics_dict.items():
    counter +=1
    print(counter)
    RID_lyrics_list = list()
    POS_lyrics_list = list()
    for lyrics in lyrics_list:
        lines = lyrics.split('<>')
        # check whether the second line gives song writer info. if so, remove the first two lines
        if lines[1][0:6] == 'Writer' or lines[1][0:7] == 'Writers':
            lines = lines[2:]
        RID_song = list()
        POS_song = list()
        for line in lines[:-1]: # because the last line is always empty in the dataset
            doc = nlp(line)
            POS_line = []
            RID_line = []
            token_line = []
            for token in doc:
                POS_line.append(token.pos_)
                token_line.append(token.text.lower())
            for token in token_line:
                if token in RID_strict_adjectives:
                    RID_line.append(reverse_RID[token])
                elif any(adj in token for adj in RID_flex_adjectives):
                    candidates = [adj for adj in RID_flex_adjectives if(token.startswith(adj))]
                    # there might be more than one candidate for each token. 'love' might match 'lov*' and 'lo*'
                    if len(candidates) == 1: # if there is only one candidate
                        RID_line.append(reverse_RID[candidates[0]+"*"])
                    elif len(candidates) >= 1: # if there are multiple candidates, select the shortest one
                        RID_line.append(reverse_RID[min(candidates, key=len)+"*"])
                    else: # if there are no matches
                        RID_line.append("NONE")
                else: # if there are no matches
                    RID_line.append("NONE")
            RID_song.append(RID_line)
            POS_song.append(POS_line)
        RID_lyrics_list.append(RID_song)
        POS_lyrics_list.append(POS_song)       
    final_artists_to_RIDsongs_dict[artist] = RID_lyrics_list
    final_artists_to_POSsongs_dict[artist] = POS_lyrics_list
    
print("Writing the final RID and POS dictionaries...")

writePickle(final_artists_to_RIDsongs_dict, "final_artists_to_RIDsongs_dict")
writePickle(final_artists_to_POSsongs_dict, "final_artists_to_POSsongs_dict")

