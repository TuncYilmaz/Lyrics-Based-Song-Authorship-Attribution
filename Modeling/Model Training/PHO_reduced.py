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
X_train_PHO1 = readPickle("cnn_data_inputs/PHO1_Keras/X_train_PHO_10_10")
X_dev_PHO1 = readPickle("cnn_data_inputs/PHO1_Keras/X_dev_PHO_10_10")
X_test_PHO1 = readPickle("cnn_data_inputs/PHO1_Keras/X_test_PHO_10_10")
y_train_PHO1 = readPickle("cnn_data_inputs/PHO1_Keras/y_train_PHO_10_10")
y_dev_PHO1 = readPickle("cnn_data_inputs/PHO1_Keras/y_dev_PHO_10_10")
y_test_PHO1 = readPickle("cnn_data_inputs/PHO1_Keras/y_test_PHO_10_10")
max_line = X_train_PHO1.shape[2]
max_song = X_train_PHO1.shape[1]
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
model.add(Dense(len(list(Artist2id.keys())), activation="softmax")) #here we need to find the length \
                                                                                    # of the potential labels


# compile all the layers    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model using the development set
history = model.fit(X_train_PHO1, y_train_PHO1, validation_data=(X_dev_PHO1, y_dev_PHO1), epochs=25, batch_size=64)


# save the predictions on the test set
predictions = model.predict(X_test_PHO1)

# save the predictions in a pickle file that is later to be used for evaluation:
writePickle(predictions, "predictions_PHO_10_10_batchnorm_25ep_64bch")

# show the accuracy of the trained model on test set
score = model.evaluate(X_test_PHO1, y_test_PHO1, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# show the model summary and save the model
model.summary()
model.save("../Saved_Models/PHO_10_10_25ep_64bch_05Drop_33filter_batchnorm")

writePickle(history,"history_PHO_10_10_25ep_64bch_05Drop_33filter_batchnorm")
