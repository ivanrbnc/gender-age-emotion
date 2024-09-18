import cv2 as cv
import numpy as np  
from keras._tf_keras.keras.models import model_from_json  
from keras._tf_keras.keras.preprocessing import image  

FACE_PROTO = "resources/opencv_face_detector.pbtxt"
FACE_MODEL = "resources/opencv_face_detector_uint8.pb"
AGE_PROTO = "resources/age_deploy.prototxt"
AGE_MODEL = "resources/age_net.caffemodel"
GENDER_PROTO = "resources/gender_deploy.prototxt"
GENDER_MODEL = "resources/gender_net.caffemodel"
FER_JSON = "resources/fer.json"
FER_WEIGHTS = "resources/fer.weights.h5"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

AGE_LIST = ['Infant', 'Child', 'Pre-Teen', 'Teenager', 'Young Adult', 'Adult', 'Middle-Aged Adult', 'Senior']
GENDER_LIST = ['Male', 'Female']
EMOTION_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

PADDING = 20

emotion_model = model_from_json(open(FER_JSON, "r").read())  
emotion_model.load_weights(FER_WEIGHTS)

def load_networks():
    face_net = cv.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return face_net, age_net, gender_net

def get_face_box(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height, frame_width = frame_opencv_dnn.shape[:2]
    blob = cv.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)

    return frame_opencv_dnn, bboxes

def get_emotion(face):
    gray_img = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
    roi_gray = cv.resize(gray_img, (48, 48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    predictions = emotion_model.predict(img_pixels, verbose=0)
    max_index = np.argmax(predictions[0])
    return EMOTION_LIST[max_index]

def main():
    face_net, age_net, gender_net = load_networks()

    cap = cv.VideoCapture(0)

    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()
        
        if not has_frame:
            print("No frame, adios!")
            break

        frame_face, bboxes = get_face_box(face_net, frame)

        for bbox in bboxes:
            face = frame[max(0, bbox[1] - PADDING):min(bbox[3] + PADDING, frame.shape[0] - 1),
                         max(0, bbox[0] - PADDING):min(bbox[2] + PADDING, frame.shape[1] - 1)]
            
            face_blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]

            emotion = get_emotion(face)

            color = (0, 0, 255) if gender == "Male" else (251, 198, 207)

            cv.putText(frame_face, gender, (bbox[0], bbox[1] - 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
            cv.putText(frame_face, age, (bbox[0], bbox[1] - 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
            cv.putText(frame_face, emotion, (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)

        cv.imshow("Age, Gender, Emotion", frame_face)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
