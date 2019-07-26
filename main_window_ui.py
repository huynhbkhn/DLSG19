import sys
import cv2
import datetime

from PyQt5.QtCore import  pyqtSlot,Qt, QPoint, QRect, QTimer, QUrl
from PyQt5.QtWidgets import  QApplication, QMainWindow, QWidget, QMessageBox, QDialogButtonBox, QFileDialog, QDialog
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QImage
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QSound
from PyQt5.uic import loadUi

import predict

# cv2 Image to qImage
def convertImage(image):
    # convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get image info
    height, width, channel = image.shape
    step = channel * width

    # create QImage from image
    qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
    return qImg

# play choach
def play():
    sound_file = "camera_shutter.wav"
    QSound.play(sound_file)

# setting ui
class settingWidget(QDialog):
    def __init__(self, parent=None):
        super(settingWidget, self).__init__(parent)
        loadUi('setting.ui', self)
        self.setFixedSize(self.size())

        # set parameter
        self.pathSave.setText(self.parent().pathSave)
        self.chkAutocapt.setChecked(True) if self.parent().autoCapt == True else self.chkAutocapt.setChecked(False)
        self.chkEmotionlbl.setChecked(True) if self.parent().emotionLbl == True else self.chkEmotionlbl.setChecked(False)
        self.chkFacebb.setChecked(True) if self.parent().faceBb == True else self.chkFacebb.setChecked(False)
        self.chkGrid.setChecked(True) if self.parent().grid == True else self.chkGrid.setChecked(False)

        # Action
        self.btnpathSave.clicked.connect(self.changePath)

        self.buttons.button(QDialogButtonBox.Save).clicked.connect(self.saveSettings)
        self.buttons.button(QDialogButtonBox.Cancel).clicked.connect(self.cancelSettings)

    def changePath(self):
        dir_ = QFileDialog.getExistingDirectory(None, 'Select a folder:')
        if dir_ != "":
            self.pathSave.setText(dir_)

    def saveSettings(self):
        self.parent().pathSave = self.pathSave.text()
        self.parent().autoCapt = True if self.chkAutocapt.isChecked() else False
        self.parent().emotionLbl = True if self.chkEmotionlbl.isChecked() else False
        self.parent().faceBb = True if self.chkFacebb.isChecked() else False
        self.parent().grid = True if self.chkGrid.isChecked() else False
        self.close()
    def cancelSettings(self):
        self.close()

# Main windown
class main(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('main.ui', self)

        # Capture camera
        self.cap = cv2.VideoCapture(0)

        # settings parameter
        self.pathSave = "D:\\"
        self.autoCapt = True
        self.emotionLbl = True
        self.faceBb = True
        self.grid = False

        # cnt frame smile
        self.smile_cnt = 0

        # create a timer
        self.timer = QTimer()
        # start timer
        self.timer.start(20)
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set capture clicked  function
        self.btnCapt.clicked.connect(self.capture)
        # set setting clicked  function
        self.btnSetting.clicked.connect(self.setting)

    # view camera
    def viewCam(self):

        # read image in BGR format
        ret, image = self.cap.read()

        # grid
        if self.grid:
            x1 = int(image.shape[0]/3)
            x2 = x1*2

            y1 = int(image.shape[1]/3)
            y2 = y1*2

            image[[x1, x2], :, :] = 100
            image[:, [y1, y2], :] = 100

        # predict happy
        image, smile = predict.facedetection(image, self.faceBb, self.emotionLbl)

        # auto capture
        if smile:
            self.smile_cnt += 1
        else:
            self.smile_cnt = 0

        if self.autoCapt and self.smile_cnt == 2*20:
            # reset smile_cnt
            self.smile_cnt = 0
            # Capture
            self.capture()
            return

        # create QImage from image
        qImg = convertImage(image)

        # show image in img_label
        self.lblCam.setPixmap(QPixmap.fromImage(qImg))

    # Capture
    def capture(self):
        # show image in lblPic
        pixmap = self.lblCam.pixmap()
        self.lblPic.setPixmap(pixmap.scaled(80, 80, Qt.KeepAspectRatio))

        # save image
        now = datetime.datetime.now()
        name = now.strftime("%Y%m%d%H%M%S%f")
        pixmap.save('{}\{}.png'.format(self.pathSave, name))

        # stop timer
        self.timer.stop()
        # read image in BGR format
        ret, image = self.cap.read()

        # overlay white nhay nhay
        overlay = image.copy()
        overlay[:, :, :] = 255
        alpha = 0.4
        image_overlay = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # create QImage from image
        qImg = convertImage(image)
        qImg_overlay = convertImage(image_overlay)

        # show image in img_label
        self.lblCam.setPixmap(QPixmap.fromImage(qImg))
        self.lblCam.setPixmap(QPixmap.fromImage(qImg_overlay))

        # start timer
        self.timer.start(20)
        play()

    # setting
    def setting(self):
        self.settings = settingWidget(self)
        self.settings.setModal(True)
        self.settings.exec()

    # Close window
    def closeEvent(self, event):
        close = QMessageBox.question(self, "Đóng", "Bạn muốn đóng chương trình không?", QMessageBox.Yes | QMessageBox.No)
        if close == QMessageBox.Yes:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # Close window
            event.accept()
        else:
            event.ignore()

app = QApplication(sys.argv)
widget = main()
widget.show()
sys.exit(app.exec_())