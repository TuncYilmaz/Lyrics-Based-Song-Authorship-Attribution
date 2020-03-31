import pickle
def writePickle( Variable, fname):
    filename = fname +".pkl"
    f = open("pickle_vars/"+filename, 'wb')
    pickle.dump(Variable, f, protocol=4)
    f.close()
def readPickle(fname):
    filename = "../pickle_vars/"+fname +".pkl" # notice the ../ addition to reach out to variables from the parent directory
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization

#import variables
X_train_POS = readPickle("cnn_data_inputs/POS_Keras/X_train_POS_reduced")
X_dev_POS = readPickle("cnn_data_inputs/POS_Keras/X_dev_POS_reduced")
X_test_POS = readPickle("cnn_data_inputs/POS_Keras/X_test_POS_reduced")
y_train_POS = readPickle("cnn_data_inputs/POS_Keras/y_train_genre")
y_dev_POS = readPickle("cnn_data_inputs/POS_Keras/y_dev_genre")
y_test_POS = readPickle("cnn_data_inputs/POS_Keras/y_test_genre")
max_line = X_train_POS.shape[2]
max_song = X_train_POS.shape[1]
Artist2id = readPickle("indexing/Artist2id")


# create a sequential model
model = Sequential()

# add model layers
model.add(Conv2D(32, kernel_size=3, padding = "same", activation="relu", input_shape=(max_song,max_line,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, padding = "same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10, activation="softmax")) #here we need to find the length \
                                                                                    # of the potential labels


# compile all the layers    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model using the development set
history = model.fit(X_train_POS, y_train_POS, validation_data=(X_dev_POS, y_dev_POS), epochs=25, batch_size=64)


# save the predictions on the test set
predictions = model.predict(X_test_POS)

# save the predictions in a pickle file that is later to be used for evaluation:
writePickle(predictions, "predictions_POS_GENRE_reduced_batchnorm_25ep_64bch")

# show the accuracy of the trained model on test set
score = model.evaluate(X_test_POS, y_test_POS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# show the model summary and save the model
model.summary()
model.save("../Saved_Models/POS_GENRE_reduced_25ep_64bch_05Drop_33filter_batchnorm")

writePickle(history,"history_POS_GENRE_reduced_25ep_64bch_05Drop_33filter_batchnorm")
