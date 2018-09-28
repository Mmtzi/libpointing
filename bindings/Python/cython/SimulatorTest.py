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
    def __init__(self, qactor, modelname, pastTimeStepsSimulator):
        super().__init__()
        # used transferfunction
        self.modelname = modelname
        self.pastTimeSteps = pastTimeStepsSimulator
        self.tf = "system:?slider=1&epp=false"
        # alias for tf name, include dpi and samplerate
        self.tf_short = "system_1_false_easy_1800_125"

        #init tf
        self.pdev = PointingDevice.create("any:")
        self.ddev = DisplayDevice.create("any:")
        self.tfct = TransferFunction.create(self.tf, self.pdev, self.ddev)

        # dt has to be adjusted to the sample rate of the mouse
        self.desiredFPS = 125

        #for stats
        self.timeMS = 0
        self.frames = 0

        #PLAYSTATES

        self.START = True
        self.PLAY = False
        self.PAUSE = False
        self.END = False

        self.pastList = []
        self.mySampleData = []
        self.actorQueue = qactor

        print("loading model...")
        if os.path.exists('ml\\models\\'+str(self.modelname)):
            try:
                self.model = load_model('ml\\models\\'+str(self.modelname))
                self.model._make_predict_function()
                print("loaded model: "+str(self.modelname))
            except:
                print("couldnt load model: "+str(self.modelname))
        else:
            print("couldnt find model: "+str(self.modelname))

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
            "thesis\\logData\\simTest\\" + self.modelname + "mouseData_timestamp" + time.strftime(
                "%Y%m%d%H%M%S") + ".csv",
            'w', newline='')

        # header of the logfile
        self.mouse_field = ['dx', 'dy', 'button', 'rx', 'ry', 'time',  'distance',
                            'targetID', 'directionX', 'directionY', 'targetX', 'targetY', 'initMouseX', 'initMouseY', 'targetSize']
            #['targetSize', 'pdx', 'pdy', 'btn']
        # mouse_field = ['dx', 'rx']

        self.writerMouse = csv.DictWriter(self.outfile, fieldnames=self.mouse_field)
        self.writerMouse.writeheader()

        self.startGame()


    def getCursorPos(self):
        return int(self.cursor.pos[0]), int(self.cursor.pos[1])

    def startGame(self):

        while True:

            events = pygame.event.get()
            for event in events:
                if self.START:

                    # initializing PointingObject and first TargetData
                    self.cursorScale = 0.2
                    # start middle
                    pyautogui.moveTo(self.screen_width / 2, self.screen_height / 2)
                    self.cursor = crosshair(pyautogui.position()[0], pyautogui.position()[1], scale=self.cursorScale,
                                            sw=self.screen_width, sh=self.screen_height)
                    # position where a stroke starts
                    self.initCursorPos = self.cursor.pos
                    pygame.mouse.set_visible(False)

                    # first pointsize
                    self.pointSize = 20
                    # first target
                    self.targetID = 1
                    # starting in the middle
                    self.oldTarget = (int(self.screen_width / 2), int(self.screen_height / 2))
                    # targetPoint with boundary conditions
                    self.targetPosition = (random.randint(0 + self.pointSize, self.screen_width - self.pointSize),
                                           random.randint(0 + self.pointSize, self.screen_height - self.pointSize))
                    # direction/distance to Target
                    self.pastDir = (
                        int(self.targetPosition[0] - self.oldTarget[0]),
                        int(self.targetPosition[1] - self.oldTarget[1]))
                    self.pastDistance = math.sqrt(pow(self.pastDir[0], 2) + pow(self.pastDir[1], 2)) - int(
                        self.pointSize / 2)
                    if len(self.pastList) == 0:
                        self.pastList = [0, 0, math.sqrt(pow(self.pastDir[0], 2) + pow(self.pastDir[1], 2)), self.pastDir[0], self.pastDir[1], self.pointSize] * self.pastTimeSteps
                    self.screen.fill((255, 255, 255))
                    self.screen.blit(self.cursor.image, self.cursor.pos)
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                        self.PLAY = True
                        self.START = False
                        self.startTime = time.time()
                    break

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                    print(event.button, "close")
                    print("TimeInSeconds: " + str(self.timeMS / 1000) + " Frames: " + str(self.frames) + " FPS: " + str(
                        int(self.frames / (self.timeMS / 1000))))
                    for dataPoint in self.mySampleData:
                        self.writerMouse.writerow({ 'dx': str(dataPoint[0]), 'dy': str(dataPoint[1]),
                                                    'button': str(dataPoint[2]),
                                                    'rx': str(dataPoint[3]), 'ry': str(dataPoint[4]),
                                                    'time': str(dataPoint[5]),'distance': str(dataPoint[6]),
                                                    'targetID': str(dataPoint[7]),
                                                    'directionX': str(dataPoint[8]), 'directionY': str(dataPoint[9]),
                                                    'targetX': str(dataPoint[10]), 'targetY': str(dataPoint[11]),
                                                    'initMouseX': str(dataPoint[12]), 'initMouseY': str(dataPoint[13]),
                                                    'targetSize': str(dataPoint[14])
                                                  })
                    print("saved data")
                    pygame.quit()
                    sys.exit()

            if self.PLAY:
                self.timeMS += self.clock.get_time()
                self.frames +=1
                pygame.display.update()
                self.screen.fill((255, 255, 255))

                #print(i, xPos,yPos)
                pygame.gfxdraw.aacircle(self.screen, self.targetPosition[0], self.targetPosition[1], max(self.pointSize * 3, self.pointSize + 35), (200, 0, 0))
                pygame.gfxdraw.aacircle(self.screen, self.targetPosition[0], self.targetPosition[1], self.pointSize, (255, 0, 0))
                pygame.gfxdraw.filled_circle(self.screen, self.targetPosition[0], self.targetPosition[1], self.pointSize, (255, 0, 0))

                #init pastDataList

                #convert pastList to a np array with one Element which has 4 columns and 20 rows (dx, dy, distanceX2Target, distanceY2Target)
                timeSeries = []
                timeSeries.append(self.pastList)
                timeSeries = np.reshape(timeSeries, (-1, 6))
                timeSeries = np.expand_dims(timeSeries, axis=0)
                timeSeries = np.array(timeSeries)
                #print(timeSeries)
                #predict next output with the data from the past 20 timesteps
                predictionsDxDy, predictButton = self.model.predict([timeSeries])

                # if predictionsDxDy[0][0] >0:
                #     pdx = int(math.ceil(predictionsDxDy[0][0]))
                # else:
                #     pdx = int(math.floor(predictionsDxDy[0][0]))
                # if predictionsDxDy[0][1] >0:
                #     pdy = int(math.ceil(predictionsDxDy[0][1]))
                # else:
                #     pdy = int(math.floor(predictionsDxDy[0][1]))

                #output
                pdx = int(round(predictionsDxDy[0][0],0))
                pdy = int(round(predictionsDxDy[0][1],0))
                self.button = round(predictButton[0][0],2)

                #useTF on output
                prx, pry = self.tfct.applyd(pdx, pdy, 0)
                #move cursor with tf output
                self.cursor.move(prx, pry)

                #if the cursor is on the border change dx/rx or. dy/ry to 0
                if (self.getCursorPos()[0] == 1 or self.getCursorPos()[0] == self.screen_width-1):
                    pdx = 0
                    prx = 0
                if (self.getCursorPos()[1] == 1 or self.getCursorPos()[1] == self.screen_height-1):
                    pdy = 0
                    pry = 0

                # 'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance',
                # 'targetID', 'directionX', 'directionY', 'targetX', 'targetY', 'initMouseX', 'initMouseY', 'targetSize'

                self.pastDir = (self.targetPosition[0] - self.getCursorPos()[0], self.targetPosition[1]- self.getCursorPos()[1])

                #create data for csv
                self.writeline = []
                self.writeline.append(pdx)
                self.writeline.append(pdy)
                self.writeline.append(self.button)
                self.writeline.append(prx)
                self.writeline.append(pry)
                self.writeline.append(time.time()-self.startTime)
                self.writeline.append(math.sqrt(pow(self.pastDir[0], 2) + pow(self.pastDir[1], 2)))
                self.writeline.append(self.targetID)
                self.writeline.append(self.pastDir[0])
                self.writeline.append(self.pastDir[1])
                self.writeline.append(self.targetPosition[0])
                self.writeline.append(self.targetPosition[1])
                self.writeline.append(self.initCursorPos[0])
                self.writeline.append(self.initCursorPos[1])
                self.writeline.append(self.pointSize)
                self.mySampleData.append(self.writeline)

                self.actorQueue.put(self.writeline)

                print(pdx, pdy, self.button, math.sqrt(pow(self.pastDir[0], 2) + pow(self.pastDir[1], 2)), self.pastDir[0], self.pastDir[1], self.pointSize)

                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.append(pdx)
                self.pastList.append(pdy)
                self.pastList.append(math.sqrt(pow(self.pastDir[0], 2) + pow(self.pastDir[1], 2)))
                self.pastList.append(self.pastDir[0])
                self.pastList.append(self.pastDir[1])
                self.pastList.append(self.pointSize)

                self.screen.blit(self.cursor.image, self.getCursorPos())

            # check targetHit
            if self.screen.get_at(self.getCursorPos()) == (255, 0, 0)and self.button >0.5:
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
                self.startTime = time.time()


            pygame.display.flip()
            self.clock.tick(self.desiredFPS)
            #print(self.clock.get_fps())
