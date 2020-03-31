# use pickle for variable reading and writing
import pickle
def writePickle( Variable, fname):
    filename = fname +".pkl"
    f = open("pickle_vars/"+filename, 'wb')
    pickle.dump(Variable, f)
    f.close()
def readPickle(fname):
    filename = "../pickle_vars/"+fname +".pkl" # notice the ../ addition to reach out to variables from the parent directory
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

# also import keras models and layers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D,Input, Flatten, MaxPooling2D, concatenate

# import variables for the POS model
X_train_POS = readPickle("cnn_data_inputs/POS_Keras/X_train_POS")
X_dev_POS = readPickle("cnn_data_inputs/POS_Keras/X_dev_POS")
X_test_POS = readPickle("cnn_data_inputs/POS_Keras/X_test_POS")
y_train_POS = readPickle("cnn_data_inputs/POS_Keras/y_train_POS")
y_dev_POS = readPickle("cnn_data_inputs/POS_Keras/y_dev_POS")
y_test_POS = readPickle("cnn_data_inputs/POS_Keras/y_test_POS")
max_line_POS = readPickle("indexing/maxPOS_linelength")
max_song_POS = readPickle("indexing/maxPOS_songlength")
Artist2id = readPickle("indexing/Artist2id") # this one is used in both components of the merged model

# import variables for the RID model
X_train_RID = readPickle("cnn_data_inputs/RID_Keras/X_train_RID")
X_dev_RID = readPickle("cnn_data_inputs/RID_Keras/X_dev_RID")
X_test_RID = readPickle("cnn_data_inputs/RID_Keras/X_test_RID")
y_train_RID = readPickle("cnn_data_inputs/RID_Keras/y_train_RID") # actually the labels are exactly the same with the POS model
y_dev_RID = readPickle("cnn_data_inputs/RID_Keras/y_dev_RID") # actually the labels are exactly the same with the POS model
y_test_RID = readPickle("cnn_data_inputs/RID_Keras/y_test_RID") # actually the labels are exactly the same with the POS model
max_line_RID = readPickle("indexing/maxRID_linelength")
max_song_RID = readPickle("indexing/maxRID_songlength")

#-----------#
#POS MODEL
#-----------#
# input shape for POS model
input_POS = Input((max_song_POS, max_line_POS, 1))
# first 2D convolutional layer for POS model
conv1_POS = Conv2D(32, (5, 5),padding = "same", activation='relu')(input_POS)
# first max pooling layer for POS model
maxp1_POS = MaxPooling2D(pool_size = (2,2))(conv1_POS)
# second 2D convolutional layer for POS model
conv2_POS = Conv2D(64, (5, 5),padding = "same", activation='relu')(maxp1_POS)
# second max pooling layer for POS model
maxp2_POS = MaxPooling2D(pool_size = (2,2))(conv2_POS)
# flatten the output for POS model
flat_POS = Flatten()(maxp2_POS)


#-----------#
#RID MODEL
#-----------#
# input shape for RID model
input_RID = Input((max_song_RID, max_line_RID, 1))
# first 2D convolutional layer for RID model
conv1_RID = Conv2D(32, (5, 5),padding = "same", activation='relu')(input_RID)
# first max pooling layer for RID model
maxp1_RID = MaxPooling2D(pool_size = (2,2))(conv1_RID)
# second 2D convolutional layer for RID model
conv2_RID = Conv2D(64, (5, 5),padding = "same", activation='relu')(maxp1_RID)
# second max pooling layer for RID model
maxp2_RID = MaxPooling2D(pool_size = (2,2))(conv2_RID)
# flatten the output for RID model
flat_RID = Flatten()(maxp2_RID)

#-----------#
#MERGE MODELS
#-----------#
merged_model = concatenate([flat_POS,flat_RID])
#merged_model = Merge(mode='concat')([flat_POS,flat_RID])

# put the merged model into a dense layer so that the output is turned into a desired shape
dense_model = Dense(len(list(Artist2id.keys())), activation='softmax')(merged_model)

# put everything into a sequential model
model = Model([input_POS, input_RID], output=dense_model)

# compile the layers
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train the model
# either one of the label sets is fine (either POS or RID, they're both the same)
model.fit([X_train_POS,X_train_RID], y_train_POS, validation_data = ([X_dev_POS,X_dev_RID], y_dev_POS),
          epochs=10, batch_size=128)

# finally get the predictions
predictions = model.predict([X_test_POS, X_test_RID])


# save the predictions in a pickle file that is later to be used for evaluation:
writePickle(predictions, "model_predictions/predictions_POS_RID")

# show the accuracy of the trained model on test set
score = model.evaluate([X_test_POS, X_test_RID], y_test_POS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# show the model summary and save the model
model.summary()
model.save("../Saved_Models/POS_RID_10ep_128bch")
