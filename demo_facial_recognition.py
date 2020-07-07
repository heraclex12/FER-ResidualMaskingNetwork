import cv2
from resnet_masking import ResMaskingNet
import numpy as np
from tensorflow.keras.optimizers import SGD
import warnings
import argparse
warnings.filterwarnings('ignore')

EMOTION_DICT = {
        0: "ANGRY",
        1: "DISGUST",
        2: "FEAR",
        3: "HAPPY",
        4: "SAD",
        5: "SURPRISE",
        6: "NEURAL"
    }


def predict_one_pic(img_path, model, face_decade):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_decade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x , y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        X = cropped_img.reshape(-1, 48, 48, 1)
        X = np.repeat(X, 3, -1)
        X = X / 255.0
        prediction = model.predict(X)
        cv2.putText(img, EMOTION_DICT[int(np.argmax(prediction))], (x, y+(h+25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                    4, cv2.LINE_AA)
        print(prediction)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_real_time(model, face_decade):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_decade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            roi_gray = gray[y: y + h, x: x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            X = cropped_img.reshape(-1, 48, 48, 1)
            X = np.repeat(X, 3, -1)
            X = X / 255.0
            prediction = model.predict(X)
            cv2.putText(frame, EMOTION_DICT[int(np.argmax(prediction))], (x, y+(h+25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 4, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Facial expression recognition for demo')
    parser.add_argument('--one-image', dest='method', default='realtime', help='predict from one image in local storage')
    args = parser.parse_args()

    print("Loading Residual Masking Network model...")
    model = ResMaskingNet(input_shape=(48, 48, 3), classes=7)
    model.compile(optimizer=SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy',
                       metrics=['accuracy'])
    model.load_weights("residual_masking_71_best.h5")
    print("========Completed load model========")

    face_decade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if args.method == 'realtime' or args.method == '':
        print("Recognizing real-time...")
        predict_real_time(model, face_decade)
    else:
        print("Recognizing one image...")
        predict_one_pic(args.method, model, face_decade)

    print("========DONE========")