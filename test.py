import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from keras.utils import normalize

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

canvas = np.zeros((280, 280, 1), np.uint8)

canvas.fill(0)

x = 0
y = 0
drawing = False


def draw(event, current_x, current_y, flags, params):
    global x, y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        canvas.fill(0)
        x = current_x
        y = current_y
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (current_x, current_y), (x, y), 255, thickness=10)
            x, y = current_x, current_y
    elif event == cv2.EVENT_LBUTTONUP:
        img = cv2.imread('input.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resize = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        plt.imshow(img_resize)
        plt.show()
        img_normalize = normalize(img_resize, axis=1)
        img_normalize = np.array(img_normalize).reshape(-1, 28, 28, 1)
        predictions = loaded_model.predict(img_normalize)
        print(np.argmax(predictions))
        drawing = False


cv2.imshow('Draw', canvas)

cv2.setMouseCallback('Draw', draw)

while (True):
    cv2.imshow('Draw', canvas)
    cv2.imwrite('input.png', canvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
