import face_recognition
import numpy as np
import cv2 as cv
import time
from PIL import Image, ImageDraw, ImageFont 

#使用CNN模型在图像列表中查找所有面孔
#使用带有CUDA的GPU 批处理的速度要快3倍  一次batch张图片
batch_size = 8
frame_list = list()
frame_count = 0
flag = 0
input_video = "/home/yipeng/videos/2.mp4"
output_video = "korean_drama.avi" 

def main():
    #要在函数中改变的值
    global frame_list
    global frame_count
    
    begin = time.time()
    #写视频
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    #文件名 编码器 fps帧率  分辨率  通道
    video_writer = cv.VideoWriter(output_video, fourcc, 25, (1280, 720), True)

    #导入图片
    an_image = face_recognition.load_image_file("安孝燮.jpg")
    che_image = face_recognition.load_image_file("车银优.jpg")
    yipeng_image = face_recognition.load_image_file("/home/yipeng/images/易鹏.jpg")
    yipeng_face_locations = face_recognition.face_locations(yipeng_image, model="cnn")
    
    #人脸编码
    an_face_encoding = face_recognition.face_encodings(an_image)[0]
    che_face_encoding = face_recognition.face_encodings(che_image)[0]
    #直接识别不出来  先进行人脸识别定位到人脸位置  再将人脸位置传入 得到人脸编码
    yipeng_face_encoding = face_recognition.face_encodings(yipeng_image, yipeng_face_locations)[0]
    
    #这是已知的人脸编码 和对应的人名
    known_face_encodings = [
                an_face_encoding, 
                che_face_encoding, 
                yipeng_face_encoding
            ]
    known_face_names = [
                #"安孝燮", 
                #"车银优", 
                #"易鹏"
                "anxiaoxie", 
                "cheyinyou", 
                "yipeng"
            ]

    video_capture = cv.VideoCapture(input_video)
    #video_capture = cv.VideoCapture(0)
    
    #是否打开video
    while video_capture.isOpened():
        #grab a frame of video
        result, frame = video_capture.read()
        
        if not result:
            break
        
        #分辨率1280 × 720 转为640 * 480   双线性插值  默认设置
        frame = cv.resize(frame, (1280, 720), cv.INTER_LINEAR)
        #opencv读的是bgr格式 改一下通道顺序为rgb 给face_recognition用
        frame = frame[:, :, ::-1]
        frame_count += 1
        frame_list.append(frame)

        if batch_size == len(frame_list):
            #使用cnn人脸检测器返回图像中人脸边界框的2d数组  GPU可以一次处理一批图像
            #批处理识别人脸 抽样次数为0
            face_locations_batch = face_recognition.batch_face_locations(frame_list, 
                                                                         number_of_times_to_upsample=0, 
                                                                         batch_size=batch_size)

            #enumerate 得到列表的下标 和 下标的值(人脸坐标  上 右 下 左)
            for frame_index_in_batch, face_locations in enumerate(face_locations_batch):
                #一帧中识别出多少张人脸
                faces_number_in_frame = len(face_locations)
                #当前帧数量
                current_frame_number = frame_count - 8 + frame_index_in_batch
                current_frame = frame_list[frame_index_in_batch]

                font = cv.FONT_HERSHEY_DUPLEX 
                name = 'UnKnown'
                name_list = list()
                flag = 0
                
                #遍历一帧中识别到的每个人脸
                for face_location in face_locations:
                    flag = 1
                    #人脸编码api中传入的人脸位置要是一个列表
                    face_list = list()
                    face_list.append(face_location)
                    #得到视频中识别出的每一帧每个人脸编码 
                    face_encodings = face_recognition.face_encodings(current_frame, face_list)[0]
                    #相匹配的两张脸之间的距离  越低表示越准确
                    face_distance = face_recognition.face_distance(known_face_encodings, face_encodings)
                    #和已知人脸编码做对比 容差tolerance 0.6是经典值 表示相匹配的两张脸之间的距离 越低越严格
                    matches = face_recognition.compare_faces(known_face_encodings, face_encodings, 0.4)
                    for index, result in enumerate(matches):
                        if result:
                            name = known_face_names[index] 
                    name_list.append(name)

                    #此时仍然是RGB格式  pil也是此格式
                    #pil_image = Image.fromarray(current_frame)
                    #pil图片上打印中文  cv的putText只支持ascii集中很小部分子集
                    #draw = ImageDraw.Draw(pil_image)
                    #字体文件路径 
                    #font = ImageFont.truetype('/usr/share/fonts/truetype/Sarai/Sarai.ttf', 20, encoding='utf-8')
                    
                    #从PIL转回rgb 给opencv使用
                    #current_frame = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
                    current_frame = current_frame[:, :, ::-1]
                    top, right, bottom, left = face_location
                    cv.rectangle(current_frame, (left, top), (right, bottom), 
                                 (0, 255, 0), 2)
                    cv.line(current_frame, (left, bottom - 35), (right, bottom - 35),
                            (0, 255, 0), 1, 8)
                    #draw.text((left + 6, bottom - 6), name, (255, 255, 255), font=font)
                    cv.putText(current_frame, name, (left + 6, bottom - 6), font,
                               1.0, (255, 255, 255), 1)

                if faces_number_in_frame == 1:
                    print("I found {0} faces in frame {1}, he is {2}".format(faces_number_in_frame, 
                                                                             current_frame_number, 
                                                                             name))
                else:
                    print("I found {0} faces in frame {1}, they are {2}".format(faces_number_in_frame, 
                                                                                current_frame_number, 
                                                                                name_list))
                
                #这里在一帧图像上 框出所有人脸了
                if not flag:
                    current_frame = current_frame[:, :, ::-1]
                cv.imshow('frame', current_frame)
                #cv.imwrite('{}.jpg'.format(current_frame_number), current_frame)
                video_writer.write(current_frame)

                if (cv.waitKey(1) & 0xFF == ord('q')):
                    break
            #每批128帧 缓存到列表  处理一批后 就清空列表
            frame_list.clear()
            end = time.time()
            print("GPU人脸识别批处理，batch_size: {0}, 耗时: ".format(batch_size), end - begin)

    video_capture.release()    
    video_writer.release()

if __name__ == '__main__':
    main()
    

