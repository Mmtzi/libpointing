import pygame
import pygame.gfxdraw
import pygame.time
import time
import csv
import math
import random
import pyautogui
from thesis.crosshair import crosshair
from keras.models import load_model
from threading import Thread
import numpy as np
import math
from queue import Queue
import os
import ctypes
import sys

from pylibpointing import PointingDevice, DisplayDevice, TransferFunction
from pylibpointing import PointingDeviceManager, PointingDeviceDescriptor

class SimTest(Thread):
    def __init__(self):
        super().__init__()
        #self.setDaemon(True)
        # used transferfunction
        self.tf = "system:?slider=1&epp=false"
        # alias for data name, include afterwards please dpi and samplerate
        self.tf_short = "system_1_false_easy_1800_125"
        self.pm = PointingDeviceManager()
        PointingDevice.idle(100)

        self.pdev = PointingDevice.create("any:")
        self.ddev = DisplayDevice.create("any:")
        self.tfct = TransferFunction.create(self.tf, self.pdev, self.ddev)
        # time
        # dt has to be adjusted to the sample rate of the mouse
        self.desiredFPS = 125

        # userSettings
        # how many sessions a user has to do
        self.session = 1
        self.sessions = 8
        # how long a session will take in Minutes
        self.sessionTargets = 100

        # sampleFlag = True
        self.timeMS = 0
        self.frames = 0
        self.START = True
        self.PLAY = False
        self.PAUSE = False
        self.END = False
        self.ISRUNNING = False
        self.pastTimeSteps = 20
        self.pastList = []
        self.mySampleData = []
        print("loading model...")
        if os.path.exists('ml\\models\\sim_conv_20dx_20dist_sizeo.h5'):
            try:
                self.model = load_model('ml\\models\\sim_conv_20dx_20dist_sizeo.h5')
                self.model._make_predict_function()
                print("loaded model")
            except:
                print("couldnt load model")
        else:
            print("couldnt find model")

    def run(self):

        pygame.init()

        self.clock = pygame.time.Clock()

        # screeninfos
        self.infoObject = pygame.display.Info()
        self.screen_width = self.infoObject.current_w
        self.screen_height = self.infoObject.current_h
        print(self.screen_width,self.screen_height)
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height], pygame.FULLSCREEN)
        pygame.display.set_caption('Simulator\' Test')

        # logfilelocation w timestamp
        self.outfile = open(
            "thesis\\logData\\simTest\\" + self.tf_short + "mouseData_timestamp" + time.strftime(
                "%Y%m%d%H%M%S") + ".csv",
            'w', newline='')

        # header of the logfile
        self.mouse_field = ['targetSize', 'pdx', 'pdy', 'btn']
        # mouse_field = ['dx', 'rx']

        self.writerMouse = csv.DictWriter(self.outfile, fieldnames=self.mouse_field)
        self.writerMouse.writeheader()

        # settings and first initialization for the study
        self.pointArray = []
        # first pointsize
        self.pointSize = 20
        # first target
        self.targetID = 1
        # starting in the middle
        self.oldTarget = (int(self.screen_width / 2), int(self.screen_height / 2))
        # targetPoint boundary conditions
        self.targetPosition = (random.randint(0 + self.pointSize, self.screen_width - self.pointSize), random.randint(0 + self.pointSize, self.screen_height - self.pointSize))
        self.pastDir = (int(self.targetPosition[0] - self.oldTarget[0]), int(self.targetPosition[1] - self.oldTarget[1]))
        self.pastDistance = math.sqrt(pow(self.pastDir[0], 2) + pow(self.pastDir[1], 2))-int(self.pointSize/2)
        # size of the
        self.cursorScale = 0.2
        # start middle
        pyautogui.moveTo(self.screen_width / 2, self.screen_height / 2)
        self.startCursorPos = pyautogui.position()
        self.initCursorPos = self.startCursorPos
        self.cursor = crosshair(self.startCursorPos[0], self.startCursorPos[1], scale=self.cursorScale, sw=self.screen_width, sh=self.screen_height)
        pyautogui.moveTo(self.startCursorPos[0],
                         self.startCursorPos[1])
        pygame.mouse.set_visible(False)


        self.startTime = time.time()

        self.pastData = [self.pointSize]
        print(self.pastData)


        self.startGame()


    def getCursorPos(self):
        #print(self.cursor.pos[0],self.cursor.pos[1])
        return int(self.cursor.pos[0]), int(self.cursor.pos[1])

    def startGame(self):

        while True:

            if self.PLAY:
                self.timeMS += self.clock.get_time()
                self.frames +=1
                pygame.display.update()
                self.screen.fill((255, 255, 255))

                #print(i, xPos,yPos)
                pygame.gfxdraw.aacircle(self.screen, self.targetPosition[0], self.targetPosition[1], max(self.pointSize * 3, self.pointSize + 35), (200, 0, 0))
                pygame.gfxdraw.aacircle(self.screen, self.targetPosition[0], self.targetPosition[1], self.pointSize, (255, 0, 0))
                pygame.gfxdraw.filled_circle(self.screen, self.targetPosition[0], self.targetPosition[1], self.pointSize, (255, 0, 0))

                #print(self.pastMouseMovement)
                #print(self.pastData + self.pastMouseMovement + self.pastDistanceList)
                if len(self.pastList) ==0:
                    self.pastList = [0,0,self.pastDir[0], self.pastDir[1]]*self.pastTimeSteps

                timeSeries= np.array(self.pastList)
                timeSeries = np.reshape(timeSeries, (-1, 4))
                #print(myInput)
                #print(np.shape(myInput))

                predictionsDxDy, predictButton = self.model.predict([timeSeries, np.array(self.pointSize)])

                if predictionsDxDy[0][0] >0:
                    pdx = int(math.ceil(predictionsDxDy[0][0]))
                else:
                    pdx = int(math.floor(predictionsDxDy[0][0]))
                if predictionsDxDy[0][1] >0:
                    pdy = int(math.ceil(predictionsDxDy[0][1]))
                else:
                    pdy = int(math.floor(predictionsDxDy[0][1]))

                #pdx = int(round(predictionsDxDy[0][0],0))
                #pdy = int(round(predictionsDxDy[0][1],0))
                self.button = round(predictButton[0][0],1)


                rx0, ry0 = self.tfct.applyd(pdx, pdy, 0)
                self.cursor.move(rx0, ry0)
                if (self.getCursorPos()[0] == 1 or self.getCursorPos()[0] == self.screen_width-1):
                    pdx = 0
                if (self.getCursorPos()[1] == 1 or self.getCursorPos()[1] == self.screen_height-1):
                    pdy = 0
                self.writeline = self.pointSize + [pdx] + [pdy] + [self.button]

                self.mySampleData.append(self.writeline)
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.append(pdx)
                self.pastList.append(pdy)
                self.pastList.append(self.targetPosition[0] - self.getCursorPos()[0])
                self.pastList.append(self.targetPosition[1]- self.getCursorPos()[1])

                #print(len(self.pastMouseMovement))
                #print(len(self.pastData))

                self.screen.blit(self.cursor.image, self.getCursorPos())

                # Mouse Click Event

            # check targetHit
            if self.screen.get_at(self.getCursorPos()) == (255, 0, 0):# and self.button >0.1:
                print("startCursorPosition:" + str(self.initCursorPos))
                self.targetID += 1
                self.oldTarget = self.targetPosition
                self.pointSize = random.randint(2, 75)
                self.targetPosition = (random.randint(0 + self.pointSize, self.screen_width - self.pointSize),
                                       random.randint(0 + self.pointSize, self.screen_height - self.pointSize))
                self.initCursorPos = self.getCursorPos()
                while (abs(self.targetPosition[0] - self.oldTarget[0]) <= self.pointSize) or (
                (abs(self.targetPosition[1] - self.oldTarget[1]) <= self.pointSize)):
                    self.targetPosition = (random.randint(0 + self.pointSize, self.screen_width - self.pointSize),
                                           random.randint(0 + self.pointSize, self.screen_height - self.pointSize))
                print("new targetX: " + str(self.targetPosition[0]) + " new targetY: " + str(self.targetPosition[1]) +
                      " distance: " + str(math.sqrt(pow(self.targetPosition[0], 2) + pow(self.targetPosition[1], 2))) +
                      " new Size: " + str(self.pointSize))
                self.pastData = [self.pointSize]
                self.startTime = time.time()
            events = pygame.event.get()
            for event in events:
                if self.START:
                    self.screen.fill((255, 255, 255))
                    self.screen.blit(self.cursor.image, self.cursor.pos)
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                        self.PLAY = True
                        self.START = False
                    break
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                    print(event.button, "close")
                    print("TimeInSeconds: " + str(self.timeMS / 1000) + " Frames: " + str(self.frames) + " FPS: " + str(
                        int(self.frames / (self.timeMS / 1000))))
                    for dataPoint in self.mySampleData:
                        # 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'targetID', 'pdx', 'pdy'
                        #print(dataPoint)
                        self.writerMouse.writerow({ 'targetSize': str(dataPoint[0]),
                                                    'pdx': str(dataPoint[1]), 'pdy': str(dataPoint[2]),
                                                    'btn': str(dataPoint[3])
                                                  })
                    print("saved data")

                    pygame.quit()
                    sys.exit()


            pygame.display.flip()
            self.clock.tick(self.desiredFPS)
                #sampleFlag=False

#if __name__ == '__main__':
    #print("Hello")