import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

np.set_printoptions(suppress=True)
results = []


def fileSelector():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    return filename


def maleModelSetup(normalizedImageArray):
    model = tensorflow.keras.models.load_model('maleModel.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    maleModelRun(model, data, normalizedImageArray)


def maleModelRun(model, data, normalizedImageArray):
    data[0] = normalizedImageArray
    prediction = model.predict(data)
    if prediction[0, 0] == 1:
        results.append("MALE")
        print("The person is MALE")
    elif prediction[0, 1] == 1:
        results.append("FEMALE")
        print("The person is FEMALE")
    else:
        results.append("NA")
        print("FAILED: ", prediction)


# def modelCaller():


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


def start():
    imagePath = fileSelector()
    image = openImage(imagePath)
    imageArray = convertImageToImageArray(image)
    showImage(image)
    normalizedImageArray = normalizeImageArray(imageArray)
    maleModelSetup(normalizedImageArray)


start()
