from lib.callbacks import LossHistory
from lib.networks import LeNet
import os

def train(x_train, y_train_cat, SIZE_BATCHS, NUM_EPOCHS,
          width, height, depth, num_classes):
    '''
    Compiling and Training the network
    '''
    ### Initialize & Compile model
    print("[INFO] Compiling model")
    model = LeNet.build(width=width, height=height, depth=depth, classes=num_classes)
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    ### Train Model
    print("[INFO] Training model")
    history = LossHistory()
    model.fit(x_train, y_train_cat, batch_size=SIZE_BATCHS, epochs=NUM_EPOCHS,
              verbose=1, callbacks=[history])

    return (model, history)

def fit(x_test, y_test_cat, model, folder):
    '''
    Fiting and Predictiong results
    '''
    ### Evaluate Model
    print("[INFO] Evaluating model")
    (loss, accuracy) = model.evaluate(x_test, y_test_cat, verbose=1)
    print("\n[INFO] Error: {:.2f}%".format((1-accuracy) * 100))
    print("[INFO] Loss: {:.2f}".format(loss))

    ### Save model
    print("[INFO] Saving model")
    model_json = model.to_json()
    path = os.path.join("../output/raw",folder)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join("../output/raw",folder,"model.json")
    with open(path, "w") as json_file:
        json_file.write(model_json)
    path = os.path.join("../output/raw",folder,"model_weights.h5")
    model.save_weights(path, overwrite=True)

    ### Predictions on Test Set
    print("[INFO] Predicting on Test Set")
    y_pred = model.predict_classes(x_test, verbose=1)

    return (y_pred, accuracy)
