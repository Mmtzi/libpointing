import pygame
import pygame.gfxdraw
import pygame.time
import time
import csv
import math
import random
import pyautogui
from thesis.crosshair import crosshair
from thesis import plotData
from ml import calcScore
from threading import Thread
import numpy as np
import threading
import sys

from pylibpointing import PointingDevice, DisplayDevice, TransferFunction
from pylibpointing import PointingDeviceManager, PointingDeviceDescriptor

class Game(Thread):
    def __init__(self, qsimu, qactor):
        super().__init__()

        # used transferfunction
        self.tf = "system:?slider=1&epp=false"
        # alias for data name, include afterwards please dpi and samplerate
        self.tf_short = "system_1_false_easy_1800_125"

        #setting up libpointing env
        self.pm = PointingDeviceManager()
        PointingDevice.idle(100)
        self.pm.addDeviceUpdateCallback(self.cb_man)
        self.pdev = PointingDevice.create("any:")
        self.ddev = DisplayDevice.create("any:")
        self.tfct = TransferFunction.create(self.tf, self.pdev, self.ddev)
        self.dpi = self.pdev.getResolution()
        self.hertz = self.pdev.getUpdateFrequency()

        # dt has to be adjusted to the sample rate in which the data was collected
        self.desiredFPS = 125

        # userSettings
        # how many sessions a user has to do
        self.session = 1
        self.sessions = 8
        # how many targets a user has to hit to complete one session
        self.sessionTargets = 100

        #for info
        self.timeMS = 0
        self.frames = 0

        #PLAYSTATES
        self.START = True
        self.PLAY = False
        self.PAUSE = False
        self.END = False

        #list for export as csv
        self.mySampleData = []
        #list for plot dx
        self.myPlotDx = []
        self.scorePlotList =[]

        self.queueSimu = qsimu
        self.queueActor = qactor

    def run(self):

        pygame.init()

        self.clock = pygame.time.Clock()

        # screeninfos
        self.infoObject = pygame.display.Info()
        self.screen_width = self.infoObject.current_w
        self.screen_height = self.infoObject.current_h
        print("Screen Resolution:"+str(self.screen_width)+","+str(self.screen_height))
        #setScreen to Fullscreen
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height], pygame.FULLSCREEN)
        pygame.display.set_caption('Fitts\' Law')

        # logfilelocation w timestamp
        self.outfile = open(
            "thesis\\logData\\adaptive\\" + self.tf_short + "mouseData_timestamp" + time.strftime(
                "%Y%m%d%H%M%S") + ".csv",
            'w', newline='')

        # header of the logfile
        self.mouse_field = ['dx', 'dy', 'button', 'rx', 'ry', 'time',  'distance',
                            'targetID', 'directionX', 'directionY', 'targetX', 'targetY', 'initMouseX', 'initMouseY', 'targetSize']
        # mouse_field = ['dx', 'rx']

        self.writerMouse = csv.DictWriter(self.outfile, fieldnames=self.mouse_field)
        self.writerMouse.writeheader()

        self.startGame()
        print("out game")
        sys.exit()

    def cb_man(desc, wasAdded):
        print(desc)
        if wasAdded:
            print("was added")
        else:
            print("was removed")

    #callback of the mouse
    def cb_fct(self, timestamp, dx0, dy0, button):
        self.newTimestamp =(timestamp / 1000000000 - self.startTime)
        rx0,ry0=self.tfct.applyd(dx0, dy0, timestamp) #timestamp unnecassary
        self.cursor.move(rx0,ry0)
        #print("%s: %d %d %d -> %.2f %.2f"%(str(newTimestamp), dx, dy, button, rx, ry ))
        direction = (self.targetPosition[0] - self.getCursorPos()[0], self.targetPosition[1] - self.getCursorPos()[1])
        distance = math.sqrt(pow(direction[0], 2) + pow(direction[1], 2))
        #'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance',
        #'directionX', 'directionY', 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'targetID'
        sampleSimu = (dx0, dy0, button, rx0, ry0, self.newTimestamp, distance, self.targetID, direction[0], direction[1], self.targetPosition[0], self.targetPosition[1],
                    self.initCursorPos[0], self.initCursorPos[1], self.pointSize)
        sampleActor = (dx0, dy0, button, rx0, ry0, self.newTimestamp, distance, self.targetID, direction[0], direction[1], self.targetPosition[0], self.targetPosition[1],
                    self.initCursorPos[0], self.initCursorPos[1], self.pointSize)
        self.myPlotDx.append(dx0)
        #add data only in playstate
        if self.PLAY:
            self.mySampleData.append(sampleSimu)
            self.queueSimu.put(sampleSimu)
            self.queueActor.put(sampleActor)
            calcScore.calcScoreOfAction(self.getCursorPos(), self.oldCursorPos, distance, self.targetPosition, self.pointSize)
            self.oldCursorPos = self.getCursorPos()
        sys.stdout.flush()

    def getCursorPos(self):
        return int(self.cursor.pos[0]), int(self.cursor.pos[1])

    def startGame(self):

        while not self.END:
            events = pygame.event.get()
            for event in events:
                if self.PAUSE and not self.PLAY and event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    self.PLAY = True
                    self.PAUSE = False
                    print("Continue")
                    break

                if self.START:

                    # initializing PointingObject and first TargetData
                    self.cursorScale = 0.2
                    # start middle
                    pyautogui.moveTo(self.screen_width / 2, self.screen_height / 2)
                    self.cursor = crosshair(pyautogui.position()[0], pyautogui.position()[1], scale=self.cursorScale,
                                            sw=self.screen_width, sh=self.screen_height)
                    # position where a stroke starts
                    self.initCursorPos = self.cursor.pos
                    self.oldCursorPos = self.cursor.pos
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
                    int(self.targetPosition[0] - self.oldTarget[0]), int(self.targetPosition[1] - self.oldTarget[1]))
                    self.pastDistance = math.sqrt(pow(self.pastDir[0], 2) + pow(self.pastDir[1], 2)) - int(
                        self.pointSize / 2)

                    self.screen.fill((255, 255, 255))
                    self.screen.blit(self.cursor.image, self.cursor.pos)

                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                        self.PLAY = True
                        self.START = False
                        self.startTime = time.time()

                    break

                #if mouse wheel pressed
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                    print(event.button, "close")
                    print("TimeInSeconds: " + str(self.timeMS / 1000) + " Frames: " + str(self.frames) + " FPS: " + str(
                        int(self.frames / (self.timeMS / 1000))))
                    #write Data in csv
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
                    #plot Histogramm of dx
                    #plotData.plotHistogramm(self.myPlotDx, self.dpi, self.hertz)
                    plotData.plotFitsDependencies(self.scorePlotList, self.dpi, self.hertz)
                    calcScore.estimate_coef(*zip(*self.scorePlotList))

                    self.END = True
                    self.PLAY = False

                # if target was hit:
                if self.PLAY and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and (
                        self.screen.get_at(self.getCursorPos()) == (255, 0, 0)):
                    self.scorePlotList.append((math.log2(math.sqrt(pow((self.targetPosition[0]-self.initCursorPos[0]), 2) + pow(self.targetPosition[1]-self.initCursorPos[1], 2))/self.pointSize*2), self.newTimestamp))
                    print("score:" +str(math.sqrt(pow((self.targetPosition[0]-self.initCursorPos[0]), 2) + pow(self.targetPosition[1]-self.initCursorPos[1], 2)) / (self.newTimestamp*self.pointSize)))
                    #print position or. new start stroke position
                    self.initCursorPos = self.getCursorPos()
                    print("startCursorPosition:"+str(self.initCursorPos))
                    self.targetID += 1
                    self.oldTarget = self.targetPosition
                    self.pointSize = random.randint(2, 75)
                    #select new target (not in the border or in the radius of the old one)
                    self.targetPosition = (random.randint(0 + self.pointSize, self.screen_width - self.pointSize), random.randint(0 + self.pointSize, self.screen_height - self.pointSize))
                    while (abs(self.targetPosition[0] - self.oldTarget[0]) <= self.pointSize) or ((abs(self.targetPosition[1] - self.oldTarget[1]) <= self.pointSize)):
                        self.targetPosition = (random.randint(0 + self.pointSize, self.screen_width - self.pointSize), random.randint(0 + self.pointSize, self.screen_height - self.pointSize))

                    print("oldtime:"+str(self.newTimestamp)+ "new targetX: " + str(self.targetPosition[0]) + " new targetY: " + str(self.targetPosition[1]) +
                          " distance: " + str(math.sqrt(pow((self.targetPosition[0]-self.initCursorPos[0]), 2) + pow(self.targetPosition[1]-self.initCursorPos[1], 2))) + " new Size: " + str(self.pointSize))
                    #start new timer for new stroke
                    self.startTime = time.time()


            # check for finish
            if self.targetID >= self.sessionTargets and self.session <= self.sessions:
                self.session+=1

                self.PAUSE = True
                self.PLAY = False

            #pause between sessions
            if self.PAUSE:
                self.screen.fill((255,255,255))

                self.pauseText = pygame.font.SysFont('Consolas', 50).render('Pause', True, pygame.color.Color("Blue"))
                self.screen.blit(self.pauseText, (100,500))
                self.sessionText = pygame.font.SysFont('Consolas', 50).render(
                    'Short Break... if you want to continue please right click', True, pygame.color.Color("Blue"))
                self.screen.blit(self.sessionText, (100,600))
                self.statText = pygame.font.SysFont('Consolas', 50).render(
                    'TimeInSeconds: ' + str(int(self.timeMS / 1000)), True, pygame.color.Color("Blue"))
                self.screen.blit(self.statText, (100,700))

            if self.PLAY:
                self.timeMS += self.clock.get_time()
                self.frames +=1
                pygame.display.update()
                self.screen.fill((255, 255, 255))
                pygame.gfxdraw.aacircle(self.screen, self.targetPosition[0], self.targetPosition[1], max(self.pointSize * 3, self.pointSize + 35), (200, 0, 0))
                pygame.gfxdraw.aacircle(self.screen, self.targetPosition[0], self.targetPosition[1], self.pointSize, (255, 0, 0))
                pygame.gfxdraw.filled_circle(self.screen, self.targetPosition[0], self.targetPosition[1], self.pointSize, (255, 0, 0))

                #TODO makes sure that you can only update cursor once each iteration of gameloop (optional)

                #mouseCallback
                self.pdev.setCallback(self.cb_fct)

                # draw new position of cursor
                self.screen.blit(self.cursor.image, self.getCursorPos())


            pygame.display.flip()
            #fpsrate
            self.clock.tick(self.desiredFPS)

        if self.END:
            print("out of pygame loop")
            return


#if __name__ == '__main__':
    #print("Hello")