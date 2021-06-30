import os
import webbrowser
from threading import Timer

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from flask import Flask, redirect, render_template

app = Flask(__name__)

results = []


def fileSelector():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    return filename


imagePath = fileSelector()


def maleModelSetup(normalizedImageArray):
    model = tensorflow.keras.models.load_model('maleModel.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    maleModelRun(model, data, normalizedImageArray)


def glassesModelSetup(normalizedImageArray):
    model = tensorflow.keras.models.load_model('glassesModel.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    glassesModelRun(model, data, normalizedImageArray)


def blondeModelSetup(normalizedImageArray):
    model = tensorflow.keras.models.load_model('blondeModel.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    blondeModelRun(model, data, normalizedImageArray)


def maleModelRun(model, data, normalizedImageArray):
    data[0] = normalizedImageArray
    prediction = model.predict(data)
    if prediction[0, 0] > 0.9:
        results.append("DEFINITELY MALE (" + str((prediction[0,0])*100) + "%)")
        print("The person is definitely MALE")
    elif prediction[0, 1] > 0.9:
        results.append("DEFINITELY FEMALE (" + str((prediction[0,1])*100) + "%)")
        print("The person is definitely FEMALE")
    elif prediction[0, 0] > 0.6:
        results.append("PROBABLY MALE (" + str((prediction[0,0])*100) + "%)")
        print("The person is probably MALE")
    elif prediction[0, 1] > 0.6:
        results.append("PROBABLY FEMALE (" + str((prediction[0,1])*100) + "%)")
        print("The person is probably FEMALE")
    else:
        results.append("NA")
        print("FAILED: ", prediction)


def glassesModelRun(model, data, normalizedImageArray):
    data[0] = normalizedImageArray
    prediction = model.predict(data)
    if prediction[0, 0] > 0.9:
        results.append("DEFINITELY WEARING GLASSES (" + str((prediction[0,0])*100) + "%)")
        print("The person is definitely WEARING GLASSES")
    elif prediction[0, 1] > 0.9:
        results.append("DEFINITELY NOT WEARING GLASSES (" + str((prediction[0,1])*100) + "%)")
        print("The person is definitely NOT WEARING GLASSES")
    elif prediction[0, 0] > 0.6:
        results.append("PROBABLY WEARING GLASSES (" + str((prediction[0,0])*100) + "%)")
        print("The person is probably WEARING GLASSES")
    elif prediction[0, 1] > 0.6:
        results.append("PROBABLY NOT WEARING GLASSES (" + str((prediction[0,1])*100) + "%)")
        print("The person is probably NOT WEARING GLASSES")
    else:
        results.append("NA")
        print("FAILED: ", prediction)


def blondeModelRun(model, data, normalizedImageArray):
    data[0] = normalizedImageArray
    prediction = model.predict(data)
    if prediction[0, 0] > 0.9:
        results.append("DEFINITELY BLONDE (" + str((prediction[0,0])*100) + "%)")
        print("The person is definitely BLONDE")
    elif prediction[0, 1] > 0.9:
        results.append("DEFINITELY NOT BLONDE (" + str((prediction[0,1])*100) + "%)")
        print("The person is definitely NOT BLONDE")
    elif prediction[0, 0] > 0.6:
        results.append("PROBABLY BLONDE (" + str((prediction[0,0])*100) + "%)")
        print("The person is probably BLONDE")
    elif prediction[0, 1] > 0.6:
        results.append("PROBABLY NOT BLONDE (" + str((prediction[0,1])*100) + "%)")
        print("The person is probably NOT BLONDE")
    else:
        results.append("NA")
        print("FAILED: ", prediction)


def modelCaller(normalizedImageArray):
    maleModelSetup(normalizedImageArray)
    glassesModelSetup(normalizedImageArray)
    blondeModelSetup(normalizedImageArray)


def openImage(imagePath):
    image = Image.open(imagePath)
    size = (224, 224)
    return ImageOps.fit(image, size, Image.ANTIALIAS)


def convertImageToImageArray(image):
    return np.asarray(image)


def showImage(image):
    image.show()


def normalizeImageArray(imageArray):
    return (imageArray.astype(np.float32) / 127.0) - 1


@app.route("/")
def mainScreen():
    # imagePath = fileSelector()
    image = openImage(imagePath)
    imageArray = convertImageToImageArray(image)
    # showImage(image)
    normalizedImageArray = normalizeImageArray(imageArray)
    modelCaller(normalizedImageArray)
    print("\n\nArray: ")
    for x in results:
        print(x)
    return redirect('/0')


@app.route("/0")
def resultsScreen():
    full_filename = os.path.basename(imagePath)
    return render_template(
        'imageView.html',
        filename=full_filename,
        aiResults=results
    )


def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run()
