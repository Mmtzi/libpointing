import os
import random
from pygame.locals import *
import pygame, sys, math
import pygame.gfxdraw
import time
import csv

# Screen Setup #################################################################
pygame.init()
scr = pygame.display.set_mode((1500, 800))
pygame.display.set_caption('Fitts\' Law')

num_of_circle = 16
circle_radius = 22  # 22, 55
distance = 500  # 125, 250, 500
current_circle = 1
pair_start = 1

# Variables for Text Input
#textinput_distance = TextInput()
#textinput_width = TextInput()
myfont1 = pygame.font.SysFont("monospace", 15)
myfont2 = pygame.font.SysFont("monospace", 13)
distance_input_activate = False
width_input_activate = False
clock = pygame.time.Clock()

# Open text file
outfile = open("timestamp" + str(time.time()) + ".csv", 'w',newline='')

outfile_field = ['num', 'W', 'D', 'ID', 'time']
writer = csv.DictWriter(outfile, fieldnames=outfile_field)
writer.writeheader()

count = 1
startingTime = 0


# Update circle ################################################################

def update_current_circle(current_circle):
    if pair_start == 1:
        if current_circle > math.ceil(num_of_circle / 2):
            return current_circle - math.ceil(num_of_circle / 2)
        else:
            return math.ceil(num_of_circle / 2) + current_circle

    # Randomly choose the start point of the next pair
    else:
        while True:
            rand = random.randint(1, num_of_circle + 1)
            if rand != current_circle:
                break
        return rand


# Game Loop ####################################################################
while True:
    pygame.display.update()
    scr.fill((255, 255, 255))

    textbox1 = pygame.draw.rect(scr, (234, 234, 234), Rect((110, 18), (70, 22)))
    textbox1_button = pygame.draw.rect(scr, (100, 50, 50), Rect((190, 18), (49, 22)))

    textbox2 = pygame.draw.rect(scr, (234, 234, 234), Rect((110, 48), (70, 22)))
    textbox2_button = pygame.draw.rect(scr, (100, 50, 50), Rect((190, 48), (49, 22)))

    # Drawing circles
    for i in range(1, num_of_circle + 1):
        pygame.gfxdraw.aacircle(scr, 750 + int(math.cos(math.pi * 2 / num_of_circle * i) * distance / 2),
                                400 + int(math.sin(math.pi * 2 / num_of_circle * i) * distance / 2),
                                circle_radius, (100, 100, 100))

    # Select a circle and make it red
    pygame.gfxdraw.filled_circle(scr, 750 + int(math.cos(math.pi * 2 / num_of_circle * current_circle) * distance / 2),
                                 400 + int(math.sin(math.pi * 2 / num_of_circle * current_circle) * distance / 2),
                                 circle_radius, (255, 0, 0))

    # Display Text
    # render text
    label1 = myfont1.render("Distance: ", 1, (0, 0, 0))
    scr.blit(label1, (20, 20))

    label2 = myfont1.render("Width: ", 1, (0, 0, 0))
    scr.blit(label2, (20, 50))

    button_text1 = myfont2.render("Enter", 1, (255, 255, 255))
    scr.blit(button_text1, (195, 22))

    button_text2 = myfont2.render("Enter", 1, (255, 255, 255))
    scr.blit(button_text2, (195, 52))

    current = myfont1.render("#" + str(count), 1, (0, 0, 0))
    scr.blit(current, (1000, 20))

    if distance_input_activate == False:
        init_distance = myfont1.render(str(int(distance)), 1, (0, 0, 0))
        scr.blit(init_distance, (115, 22))

    if width_input_activate == False:
        init_width = myfont1.render(str(circle_radius), 1, (0, 0, 0))
        scr.blit(init_width, (115, 50))

    #scr.blit(textinput_distance.get_surface(), (115, 22))
    #scr.blit(textinput_width.get_surface(), (115, 50))

    # Mouse Click Event
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = scr.get_at(pygame.mouse.get_pos()) == (255, 0, 0)
            if click == 1:
                current_circle = update_current_circle(current_circle)
                if pair_start == 1:
                    print("Log the time")
                    # Starting time
                    startingTime = time.time()
                    pair_start = 0
                else:
                    print("Random starting point!")
                    # Log: Finishing time
                    writer.writerow({'num': str(count),'W': str(circle_radius), 'D': str(distance),
                                     'ID': str(math.log2(distance / circle_radius + 1)),
                                     'time': str((time.time() - startingTime))})
                    count += 1
                    pair_start = 1

            if textbox1.collidepoint(pygame.mouse.get_pos()):
                distance_input_activate = True
                width_input_activate = False
                print("Distance input start")

            if textbox1_button.collidepoint(pygame.mouse.get_pos()):
                distance = 600
                #distance = float(textinput_distance.get_text())
                distance_input_activate = False
                count = 1
                print("Distance input finish")

            if textbox2.collidepoint(pygame.mouse.get_pos()):
                width_input_activate = True
                distance_input_activate = False
                print("Width input start")

            if textbox2_button.collidepoint(pygame.mouse.get_pos()):
                circle_radius = 20
                #circle_radius = int(textinput_width.get_text())
                width_input_activate = False
                count = 1
                print("Width input finish")

        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # Text input
    if distance_input_activate:
        pass
        # Feed it with events every frame
        #textinput_distance.update(events)

    if width_input_activate:
        pass
        # Feed it with events every frame
        #textinput_width.update(events)
################################################################################