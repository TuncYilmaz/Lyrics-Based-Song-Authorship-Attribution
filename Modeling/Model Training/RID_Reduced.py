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
X_train_RID = readPickle("cnn_data_inputs/RID_Keras/X_train_RID_reduced")
X_dev_RID = readPickle("cnn_data_inputs/RID_Keras/X_dev_RID_reduced")
X_test_RID = readPickle("cnn_data_inputs/RID_Keras/X_test_RID_reduced")
y_train_RID = readPickle("cnn_data_inputs/RID_Keras/y_train_RID_reduced")
y_dev_RID = readPickle("cnn_data_inputs/RID_Keras/y_dev_RID_reduced")
y_test_RID = readPickle("cnn_data_inputs/RID_Keras/y_test_RID_reduced")
max_line = X_train_RID.shape[2]
max_song = X_train_RID.shape[1]
Artist2id = readPickle("indexing/Artist2id")


# create a sequential model
model = Sequential()

# add model layers
model.add(Conv2D(32, kernel_size=1, padding = "same", activation="relu", input_shape=(max_song,max_line,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Conv2D(64, kernel_size=1, padding = "same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(len(list(Artist2id.keys())), activation="softmax")) #here we need to find the length \
                                                                                    # of the potential labels


# compile all the layers    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model using the development set
history = model.fit(X_train_RID, y_train_RID, validation_data=(X_dev_RID, y_dev_RID), epochs=25, batch_size=64)


# save the predictions on the test set
predictions = model.predict(X_test_RID)

# save the predictions in a pickle file that is later to be used for evaluation:
writePickle(predictions, "predictions_RID_reduced_batchnorm_25ep_64bch")

# show the accuracy of the trained model on test set
score = model.evaluate(X_test_RID, y_test_RID, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# show the model summary and save the model
model.summary()
model.save("../Saved_Models/RID_reduced_25ep_64bch_05Drop_11filter_batchnorm")

writePickle(history,"history_RID_reduced_25ep_64bch_05Drop_11filter_batchnorm")
