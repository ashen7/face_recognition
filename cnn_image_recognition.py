import cv2 as cv
import face_recognition
import time

image = face_recognition.load_image_file("/home/yipeng/images/卷仔.jpg")
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

print("I found {} face(s) in this photograph.".format(len(face_locations)))

image1 = image * 1
image1[:, :, 0] = image[:, :, 2]
image1[:, :, 2] = image[:, :, 0]
begin = time.time()
for face_location in face_locations:
    # Print the location of each face in this image
    top, right, bottom, left = face_locations[0]
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    cv.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
    
end = time.time()
print("cnn卷积神经网络耗时：", end - begin)
cv.imshow("yipeng", image1)
cv.waitKey(0)

