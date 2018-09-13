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

class ActorTrain(Thread):
    def __init__(self, queueU, queueS):
        super().__init__()


    def run(self):
        pass
