import cv2
import tensorflow.keras as keras
import numpy as np
def get_results(img_path):
    cartoon_model_classes = ['cartoon/edited', 'real']
    watermark_model_classes = ['no watermark', 'watermarked']
    model = keras.models.load_model("final_model")
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.reshape((-1, 300, 300, 3))
    real, watermark = model.predict(img)
    return [cartoon_model_classes[int(np.around(real)[0])], watermark_model_classes[int(np.around(watermark)[0])]]



def get_liveliness(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 300, 300, 3))
    classes = ['lively', 'not_lively']

    model = keras.models.load_model("liveliness_detection")

    pred = model.predict(img)
    pred = int(np.around(pred)[0])

    return classes[pred]