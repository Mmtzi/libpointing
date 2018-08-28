import pygame.image

class crosshair:

    def __init__(self, x , y, scale):
        self.image = pygame.image.load('thesis\\cursor.png')
        self.size = self.image.get_rect().size
        self.image = pygame.transform.scale(self.image, (int(self.size[0]*scale), int(self.size[1]*scale)))
        self.pos = self.image.get_rect().move(x, y)
        print("init cursor" + str(self.pos))

    def move(self, dx, dy):
        self.pos = self.pos.move(dx,dy)