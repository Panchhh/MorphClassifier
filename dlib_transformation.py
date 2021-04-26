import cv2
import dlib


def patch_image(image, x1, x2, y1, y2, shape=None):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\models_dlib\shape_predictor_68_face_landmarks.dat')

    detect = detector(image, 1)
    predicted = predictor(image, detect[0])

    x1, x2 = predicted.part(x1).x, predicted.part(x2).x
    y1, y2 = predicted.part(y1).y, predicted.part(y2).y
    x_diff = x2 - x1
    y_diff = y2 - y1
    if x_diff > y_diff:
        diff = x_diff - y_diff
        margin = diff // 2
        y1_new = max(y1 - margin, 0)
        compensate = max(-(y1 - margin), 0)
        y2_new = min(y2 + margin, image.shape[0]) + compensate
        out = image[y1_new:y2_new, x1:x2]
    else:
        diff = y_diff - x_diff
        margin = diff // 2
        x1_new = max(x1 - margin, 0)
        compensate = max(-(x1 - margin), 0)
        x2_new = min(x2 + margin, image.shape[1]) + compensate
        out = image[y1:y2, x1_new:x2_new]

    if shape is not None:
        return cv2.resize(out, shape)
    else:
        return out


def lefteye(shape):
    return lambda data: patch_image(data, 17, 21, 19, 41, shape)


def righteye(shape):
    return lambda data: patch_image(data, 22, 26, 24, 47, shape)


def botheyes(shape):
    return lambda data: patch_image(data, 17, 26, 19, 41, shape)


def innerface(shape):
    return lambda data: patch_image(data, 36, 45, 21, 57, shape)


def mouth(shape):
    return lambda data: patch_image(data, 3, 13, 30, 8, shape)


def mouth_only(shape):
    return lambda data: patch_image(data, 48, 54, 50, 57, shape)


def nose(shape):
    return lambda data: patch_image(data, 31, 35, 28, 33, shape)


def dlib_face_crop(perc=.2):
    def crop_face(data):
        detector = dlib.get_frontal_face_detector()
        face_detection_bbs = detector(data, 1)

        x = int((face_detection_bbs[0].right() - face_detection_bbs[0].left()) * perc)
        y = int((face_detection_bbs[0].bottom() - face_detection_bbs[0].top()) * perc)
        # face = img[face_detection_bbs[0].top():face_detection_bbs[0].bottom(),
        #        face_detection_bbs[0].left():face_detection_bbs[0].right()]
        if face_detection_bbs[0].top() - int(y / 2) > 0:
            new_top = face_detection_bbs[0].top() - int(y / 2)
        else:
            new_top = face_detection_bbs[0].top()
        if face_detection_bbs[0].bottom() + int(y / 2) < data.shape[0]:
            new_bottom = face_detection_bbs[0].bottom() + int(y / 2)
        else:
            new_bottom = face_detection_bbs[0].bottom()
        if face_detection_bbs[0].left() - int(x / 2) > 0:
            new_left = face_detection_bbs[0].left() - int(x / 2)
        else:
            new_left = face_detection_bbs[0].left()
        if face_detection_bbs[0].right() + int(x / 2) < data.shape[1]:
            new_right = face_detection_bbs[0].right() + int(x / 2)
        else:
            new_right = face_detection_bbs[0].right()
        return data[new_top:new_bottom, new_left:new_right]

    return lambda data: crop_face(data)
