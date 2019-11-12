import cv2 as cv
import face_recognition
import numpy as np

def main():
    #本地摄像头
    video_capture = cv.VideoCapture(0)
    #加载样本图片并学习如何识别它
    yipeng_image = face_recognition.load_image_file(file='/home/yipeng/images/yipeng.jpg')
    yipeng_face_encoding = face_recognition.face_encodings(yipeng_image)[0]

    known_face_encodings = [
        yipeng_face_encoding,
    ]
    known_face_names = [
        '易鹏'
    ]
    while video_capture.isOpened():
        result, frame = video_capture.read()
        # 将每一帧调整为4分之一尺寸 以加快人脸识别处理
        small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
        #bgr 转 rgb
        small_frame = small_frame[:, :, ::-1]
        # 人脸检测器 神经网络检测
        face_locations = face_recognition.face_locations(img=small_frame,
                                                         number_of_times_to_upsample=1,
                                                         model='cnn')
        #得到这张图片 给定人脸坐标的人脸编码
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index =


if __name__ == '__main__':
    main()