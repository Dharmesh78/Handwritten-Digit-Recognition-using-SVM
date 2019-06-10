# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HDRapp.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from PyQt5.QtCore import  Qt
from PyQt5.QtGui import QImage,  QPixmap
from PyQt5.QtWidgets import ( QApplication, QPushButton, QFileDialog, QLabel,
                             QMainWindow,  QScrollArea, QLineEdit)


class Ui_Dialog(QMainWindow):

    def __init__(self):
        super(Ui_Dialog, self).__init__()
        self.fileName = ""
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.scrollArea = QScrollArea()

        self.scrollArea.setWidget(self.imageLabel)

        self.setCentralWidget(self.scrollArea)



        #creation of  button for select
        self.button = QPushButton('select', self)
        self.button.move(200, 500)
        self.button.clicked.connect(self.on_click1)
        self.button.setStyleSheet("background-color: yellow")


        #creation of  button for predict
        self.button2 = QPushButton('predict', self)
        self.button2.move(880, 500)
        self.button2.clicked.connect(self.on_click2)
        self.button2.setStyleSheet("background-color: red")


        #creation of textbox for displaying output
        self.textbox = QLineEdit(self)
        self.textbox.move(800, 20)
        self.textbox.resize(880, 400)



        self.setWindowTitle("HANDWRITTEN DIGIT RECOGNITION                    -----DharmeshDiwakarUtkarshAman")
        self.resize(1400, 600)

        # Set window background color
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.lightGray)
        self.setPalette(p)

    def on_click1(self):

        #for opening a file dialog Box to browse the image file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                       "All Files (*);;Python Files (*.py)", options=options)

        print(self.fileName)
        image = QImage(self.fileName)

        #set the image on the image label
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.scaleFactor = 1.0
        self.imageLabel.adjustSize()

    def on_click2(self):


        # Loading the classifier
        clf = joblib.load("digits_cls.pkl")

        # Read the input image
        im = cv2.imread(self.fileName)

        # Convert to grayscale and apply Gaussian filtering
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

        # Threshold the image
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        _, ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        ll = []
        # For each rectangular region, calculate HOG features and predict
        # the digit using Linear SVM.
        for rect in rects:
            # Draw the rectangles

            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)
            # Make the rectangular region around the digit
            leng = int(rect[3])
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
            # Resize the image
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

            #getting the predicted digit in numpy 1 dimensional array
            nbr = clf.predict(np.array([roi_hog_fd], 'float64'))


            #displaying the predicted digit above the rectangular region
            cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)


            print(nbr, type(nbr), nbr.shape)
            #converting the 1d numpy array to list and getting the first element
            a = nbr.tolist()[0]

            #adding each predicted digit to list
            ll.append(a)


        print(ll)

        #converting the list to string
        string = ''.join(map(str, ll))
        print(string)

        from PIL import Image

        #converting the numpy array to image
        img = Image.fromarray(im, 'RGB')
        print(img)
        img.show()

        #displaying output string in the textbox widget
        self.textbox.setText(string)
        f = self.textbox.font()
        f.setPointSize(90)  # sets the size to 90
        self.textbox.setFont(f)


if __name__ == '__main__':
    import sys

    # PyQt5 application must create an application object. The sys.argv parameter is a list of arguments from a command line
    # This calls the constructor of the C++ class QApplication. It uses sys.argv  to initialize the QT application
    app = QApplication(sys.argv)
    hdrappobj = Ui_Dialog()
    hdrappobj.show()

    sys.exit(app.exec_())