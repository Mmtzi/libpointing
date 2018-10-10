import pygame.image

class crosshair:

    def __init__(self, x , y, scale, sw, sh):
        self.image = pygame.image.load('thesis\\cursor.png')
        self.size = self.image.get_rect().size
        self.image = pygame.transform.scale(self.image, (int(self.size[0]*scale), int(self.size[1]*scale)))
        self.pos = self.image.get_rect().move(x, y)
        self.screenwidth = sw
        self.screenheight = sh
        print("init cursor" + str(self.pos))

    def move(self, rx, ry):
        if self.pos[0]+rx < 0:
            print(self.pos[0])
            self.pos[0] = 1
        elif self.pos[0]+rx > self.screenwidth:
            print(self.pos[0])
            self.pos[0] = self.screenwidth-1
        else:
            self.pos[0] = self.pos[0] + rx
        if self.pos[1]+ry < 0:
            print(self.pos[1])
            self.pos[1] = 1
        elif self.pos[1]+ry > self.screenheight:
            print(self.pos[1])
            self.pos[1] = self.screenheight-1
        else:
            self.pos[1] = self.pos[1] + ry
