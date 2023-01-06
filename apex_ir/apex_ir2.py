from mss.windows import MSS as mss
from numba import jit
import numpy as np
import mss.tools
import cv2 as cv
import time

class Apex_IR:

    weightsfile = r'C:\Users\Neil\PycharmProjects\apex_ir\dummiemodel\dummiedetector.weights'
    configfile = r'C:\Users\Neil\PycharmProjects\apex_ir\dummiemodel\dummieconfig.cfg'
    namesfile = r'C:\Users\Neil\PycharmProjects\apex_ir\dummiemodel\obj.names'
    scdimensions = {'top': 0, 'left': 0, 'width': 1439, 'heights': 899}
    sct = mss.mss()

    def __init__(self):

        self.net = cv.dnn.readNet(self.weightsfile, self.configfile)
        self.classes = []

        with open(self.namesfile, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.layer_names = self.net.getLayerNames()
        self.outputLayers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))



    def ds_pure(self):

        cv.namedWindow('scapture', cv.WINDOW_NORMAL)

        while True:

            last_time = time.time()
            img = np.array(self.sct.grab({'top': 0, 'left': 0, 'width': 2560, 'height': 1440}))
            cv.imshow('scapture', img)
            print(f'FPS: {1 / (time.time() - last_time)}')

            if cv.waitKey(25) & 0xFF == ord('l'):
                cv.destroyAllWindows()
                break


    def run(self):

        cv.namedWindow('scapture', cv.WINDOW_NORMAL)

        while True:

            last_time = time.time()
            img = self.image_logic()
            cv.imshow('scapture', img[0])
            print(img[1:3])
            print(f'FPS: {1 / (time.time() - last_time)}')

            if cv.waitKey(25) & 0xFF == ord('l'):
                cv.destroyAllWindows()
                break



    def image_logic(self):

        self.center_x = None
        self.center_y = None

        self.image = np.array(self.sct.grab({'top': 0, 'left': 0, 'width': 2560, 'height': 1440}))
        self.image = np.flip(self.image[:, :, :3], 2)
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        # self.image = cv.resize(self.image, None, fx=0.4, fy=0.4)
        self.image = cv.resize(self.image, None, fx=1, fy=1)
        self.height, self.width, self.channels = self.image.shape

        self.blob = cv.dnn.blobFromImage(self.image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(self.blob)
        self.outs = self.net.forward(self.outputLayers)

        self.class_ids = []
        self.confidences = []
        self.boxes = []

        for out in self.outs:

            for detection in out:

                self.scores = detection[5:]
                self.class_id = np.argmax(self.scores)
                self.confidence = self.scores[self.class_id]

                if self.confidence > 0.5:
                    self.center_x = int(detection[0] * self.width)
                    self.center_y = int(detection[1] * self.height)
                    self.w = int(detection[2] * self.width)
                    self.h = int(detection[3] * self.height)

                    cv.circle(self.image, (self.center_x, self.center_y), 10, (0, 255, 0), 2)

                    self.x = int(self.center_x - self.w / 2)
                    self.y = int(self.center_y - self.h / 2)

                    self.boxes.append([self.x, self.y, self.w, self.h])
                    self.confidences.append(float(self.confidence))
                    self.class_ids.append(self.class_id)

        self.indexes = cv.dnn.NMSBoxes(self.boxes, self.confidences, 0.4, 0.6)

        il_varlist = [self.image, self.center_x, self.center_y]

        return il_varlist
