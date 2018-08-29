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

    def move(self, dx, dy):
        self.pos = self.pos.move(dx,dy)
        if self.pos[0] < 0:
            self.pos[0] = 1
        if self.pos[0] > self.screenwidth:
            self.pos[0] = self.screenwidth-1
        if self.pos[1] < 0:
            self.pos[1] = 1
        if self.pos[1] > self.screenheight:
            self.pos[1] = self.screenheight-1