
import sys
import cv2
import numpy
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image, ImageDraw, ImageFont

from Demo import Camera_reader
from train_model import Model






# 解决cv2.putText绘制中文乱码
def cv2ImgAddText(img2, text, left, top, textColor=(0, 0, 255), textSize=20):
    if isinstance(img2, numpy.ndarray):
        img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img2)
    fontStyle = ImageFont.truetype(r"C:\WINDOWS\FONTS\MSYH.TTC", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)
    return cv2.cvtColor(numpy.asarray(img2), cv2.COLOR_RGB2BGR)

# 新的名字列表
new_names = ["景甜", "王祖贤"]




class FaceDetectionApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("人脸检测应用")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.upload_button = QPushButton("图片识别")
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setFixedSize(779, 50)

        self.camera_button = QPushButton("摄像头识别")
        self.camera_button.clicked.connect(self.start_camera_detection)
        self.camera_button.setFixedSize(779, 50)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(779, 500)

        self.result_label = QLabel("识别结果: ")
        self.result_label.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.camera_button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.result_label)

        self.central_widget.setLayout(self.layout)

        self.model = Model()
        self.model.load()


    def upload_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)", options=options)

        if file_name:
            image = cv2.imread(file_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
            faces = face_cascade.detectMultiScale(gray, 1.35, 5)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi = gray[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_LINEAR)

                    label, prob = self.model.predict(roi)

                    if prob > 0.7:
                        show_name = new_names[label]
                        res = f"识别为: {show_name}, 概率: {prob:.2f}"
                    else:
                        show_name = "陌生人"
                        res = "抱歉，未识别出该人！请尝试增加数据量来训练模型！"

                    frame = cv2ImgAddText(image, show_name, x + 5, y - 30)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

                    cv2.imwrite('prediction.jpg', frame)
                    self.result = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

                    self.QtImg = QImage(
                        self.result.data, self.result.shape[1], self.result.shape[0], QImage.Format_RGB32)
                    self.image_label.setPixmap(QPixmap.fromImage(self.QtImg))
                    self.image_label.setScaledContents(True)  # 自适应界面大小

                    self.result_label.setText(res)
            else:
                self.result_label.setText("未检测到人脸")


    def start_camera_detection(self):
        self.camera = Camera_reader()
        self.camera.build_camera()


class Camera_reader(object):
    def __init__(self):
        self.model = Model()
        self.model.load()
        self.img_size = 128

    def build_camera(self):
        face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
        cameraCapture = cv2.VideoCapture(0)
        success, frame = cameraCapture.read()
        while success and cv2.waitKey(1) == -1:
            success, frame = cameraCapture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                ROI = gray[x:x + w, y:y + h]
                ROI = cv2.resize(ROI, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                label, prob = self.model.predict(ROI)

                if prob > 0.7:
                    show_name = new_names[label]
                else:
                    show_name = "陌生人"
                frame = cv2ImgAddText(frame, show_name, x + 5, y - 30)

                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Camera", frame)
        else:
            cameraCapture.release()
            cv2.destroyAllWindows()





if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())
