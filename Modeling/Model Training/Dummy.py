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


from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D

#import variables
X_train_dummy = readPickle("cnn_data_inputs/dummy_Keras/X_train_dummy")
X_dev_dummy = readPickle("cnn_data_inputs/dummy_Keras/X_dev_dummy")
X_test_dummy = readPickle("cnn_data_inputs/dummy_Keras/X_test_dummy")
y_train_dummy = readPickle("cnn_data_inputs/dummy_Keras/y_train_dummy")
y_dev_dummy = readPickle("cnn_data_inputs/dummy_Keras/y_dev_dummy")
y_test_dummy = readPickle("cnn_data_inputs/dummy_Keras/y_test_dummy")
max_line = 20
max_song = 100
Artist2id = readPickle("indexing/Artist2id")


# create a sequential model
model = Sequential()

# add model layers
model.add(Conv2D(32, kernel_size=3, padding = "same", activation="relu", input_shape=(max_song,max_line,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, padding = "same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(len(list(Artist2id.keys())), activation="softmax")) #here we need to find the length \
                                                                                    # of the potential labels


# compile all the layers    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model using the development set
model.fit(X_train_dummy, y_train_dummy, validation_data=(X_dev_dummy, y_dev_dummy), epochs=25, batch_size=128)


# save the predictions on the test set
predictions = model.predict(X_test_dummy)

# save the predictions in a pickle file that is later to be used for evaluation:
writePickle(predictions, "predictions_dummy_33")

# show the accuracy of the trained model on test set
score = model.evaluate(X_test_dummy, y_test_dummy, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# show the model summary and save the model
model.summary()
model.save("../Saved_Models/dummy_25ep_128bch_025Drop_33filter")
