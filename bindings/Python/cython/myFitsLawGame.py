import pygame
import pygame.gfxdraw
import pygame.time
import time
import csv
import math
import random
import pyautogui
from bindings.Python.cython.thesis.crosshair import crosshair
from threading import Thread
import numpy as np

import sys

from pylibpointing import PointingDevice, DisplayDevice, TransferFunction
from pylibpointing import PointingDeviceManager, PointingDeviceDescriptor

class Game(Thread):
    def __init__(self, q):
        super().__init__()
        #self.setDaemon(True)
        # used transferfunction
        self.tf = "system:?slider=1&epp=true"
        # alias for data name, include afterwards please dpi and samplerate
        self.tf_short = "system_0_true_easy_1800_125"

        self.pm = PointingDeviceManager()
        PointingDevice.idle(100)
        self.pm.addDeviceUpdateCallback(self.cb_man)

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

        self.mySampleData = []

        self.dataQueue = q

    def run(self):

        pygame.init()

        self.clock = pygame.time.Clock()

        # screeninfos
        self.infoObject = pygame.display.Info()
        self.screen_width = self.infoObject.current_w
        self.screen_height = self.infoObject.current_h
        print(self.screen_width,self.screen_height)
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height], pygame.FULLSCREEN)
        pygame.display.set_caption('Fitts\' Law')

        # logfilelocation w timestamp
        self.outfile = open(
            "thesis\\logData\\adaptive\\" + self.tf_short + "mouseData_timestamp" + time.strftime(
                "%Y%m%d%H%M%S") + ".csv",
            'w', newline='')

        # header of the logfile
        self.mouse_field = ['dx', 'dy', 'rx', 'ry', 'button', 'time',  'distance',
                            'directionX', 'directionY', 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'targetID']
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
        self.cursor = crosshair(self.startCursorPos[0], self.startCursorPos[1], scale=self.cursorScale)
        pyautogui.moveTo(self.startCursorPos[0],
                         self.startCursorPos[1])
        pygame.mouse.set_visible(False)


        self.startTime = time.time()

        self.startGame()


    def cb_man(desc, wasAdded):
        print(desc)
        if wasAdded:
            print("was added")
        else:
            print("was removed")

    #callback of the mouse
    def cb_fct(self, timestamp, dx0, dy0, button):
        newTimestamp =(timestamp / 1000000000 - self.startTime)
        rx0,ry0=self.tfct.applyd(dx0, dy0, timestamp)
        self.cursor.move(rx0,ry0)
        #print("%s: %d %d %d -> %.2f %.2f"%(str(newTimestamp), dx, dy, button, rx, ry ))
        direction = (self.targetPosition[0] - self.getCursorPos()[0], self.targetPosition[1] - self.getCursorPos()[1])
        distance = math.sqrt(pow(direction[0], 2) + pow(direction[1], 2))-int(self.pointSize/2)
        #'dx', 'dy', 'rx', 'ry', 'button', 'time', 'distance',
        #'directionX', 'directionY', 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'targetID'
        mySample = (dx0, dy0, rx0, ry0, button, newTimestamp, distance, direction[0], direction[1], self.targetPosition[0], self.targetPosition[1],
                    self.pointSize, self.initCursorPos[0], self.initCursorPos[1], self.targetID)
        #print(mySample)
        self.mySampleData.append(mySample)
        #print(mySample)
        self.dataQueue.put(mySample)

        self.pastDir = (direction[0],direction[1])
        self.pastDistance = distance
        sys.stdout.flush()


    def getCursorPos(self):
        #print(self.cursor.pos[0],self.cursor.pos[1])
        return int(self.cursor.pos[0]), int(self.cursor.pos[1])

    def stop(self):
        self._stop()

    def startGame(self):

        while True:
            events = pygame.event.get()
            for event in events:

                if self.PLAY and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and (
                        self.getCursorPos()[0] <= 0 or self.getCursorPos()[0] >= self.screen_width or self.getCursorPos()[1] <= 0 or
                        self.getCursorPos()[1] >= self.screen_height):
                    break

                if self.PAUSE and not self.PLAY and event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    self.PLAY = True
                    self.PAUSE = False
                    print("Continue")
                    break

                if self.PLAY and not self.PAUSE and event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    self.PAUSE = True
                    self.PLAY = False
                    print("Pause")
                    break

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
                        # easy
                        # writerMouse.writerow({'dx': str(dataPoint[1]), 'rx': str(dataPoint[3])})
                        # advanced
                        # 'dx', 'dy', 'rx', 'ry', 'button', 'time', 'distance',
                        # 'directionX', 'directionY', 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'targetID'
                        self.writerMouse.writerow({ 'dx': str(dataPoint[0]), 'dy': str(dataPoint[1]),
                                                    'rx': str(dataPoint[2]), 'ry': str(dataPoint[3]),
                                                    'button': str(dataPoint[4]), 'time': str(dataPoint[5]),
                                                    'distance': str(dataPoint[6]),
                                                    'directionX': str(dataPoint[7]), 'directionY': str(dataPoint[8]),
                                                    'targetX': str(dataPoint[9]), 'targetY': str(dataPoint[10]),
                                                    'targetSize': str(dataPoint[11]),
                                                    'initMouseX': str(dataPoint[12]),'initMouseY': str(dataPoint[13]),
                                                    'targetID': str(dataPoint[14])
                                                  })
                    print("saved data")

                    pygame.quit()
                    sys.exit()
                if self.PLAY and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and (
                        self.screen.get_at(self.getCursorPos()) == (255, 0, 0)):
                    print("startCursorPosition:"+str(self.initCursorPos))
                    self.targetID += 1
                    self.oldTarget = self.targetPosition
                    self.pointSize = random.randint(2, 75)
                    self.targetPosition = (random.randint(0 + self.pointSize, self.screen_width - self.pointSize), random.randint(0 + self.pointSize, self.screen_height - self.pointSize))
                    self.initCursorPos = self.getCursorPos()
                    while (abs(self.targetPosition[0] - self.oldTarget[0]) <= self.pointSize) or ((abs(self.targetPosition[1] - self.oldTarget[1]) <= self.pointSize)):
                        self.targetPosition = (random.randint(0 + self.pointSize, self.screen_width - self.pointSize), random.randint(0 + self.pointSize, self.screen_height - self.pointSize))
                    print(event.button, "new targetX: " + str(self.targetPosition[0]) + " new targetY: " + str(self.targetPosition[1]) +
                          " distance: " + str(math.sqrt(pow(self.targetPosition[0], 2) + pow(self.targetPosition[1], 2))) + " new Size: " + str(self.pointSize))
                    self.startTime = time.time()


            # check for finish
            if self.PLAY and (self.targetID) >= self.sessionTargets and self.session <= self.sessions:
                self.session+=1

                self.PAUSE = True
                self.PLAY = False

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

                #print(i, xPos,yPos)
                pygame.gfxdraw.aacircle(self.screen, self.targetPosition[0], self.targetPosition[1], max(self.pointSize * 3, self.pointSize + 35), (200, 0, 0))
                pygame.gfxdraw.aacircle(self.screen, self.targetPosition[0], self.targetPosition[1], self.pointSize, (255, 0, 0))
                pygame.gfxdraw.filled_circle(self.screen, self.targetPosition[0], self.targetPosition[1], self.pointSize, (255, 0, 0))

                #makes sure that you can only update cursor once each iteration of gameloop
                #if sampleFlag:
                self.pdev.setCallback(self.cb_fct)
                    #print("mouse update")
                #else:
                    #sampleFlag= True

                self.screen.blit(self.cursor.image, self.getCursorPos())

                # Mouse Click Event

            pygame.display.flip()
            self.clock.tick(self.desiredFPS)
                #sampleFlag=False

#if __name__ == '__main__':
    #print("Hello")