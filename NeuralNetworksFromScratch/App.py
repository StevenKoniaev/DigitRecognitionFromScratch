import pygame, sys
from keras.datasets import mnist
from pygame.locals import *
import numpy as np
from pygame import QUIT, MOUSEBUTTONUP, MOUSEBUTTONDOWN, MOUSEMOTION
import cv2
import NeuralNetwork
import matplotlib
import matplotlib.pyplot as plt

def menu():
    print()
    print("[1] Train Model")
    print("[2] Overwrite & Save Model")
    print("[3] Load Currently Saved Model")
    print("[4] Open Testing Tool")
    print("[0] Quit")

def app():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    NN.test(test_X, test_y)

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    boundary = 5

    IMAGESAVE = False
    Labels = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
              9: "Nine"}

    WINDOWSIZEX = 400
    WINDOWSIZEY = 400

    # Start pygame
    pygame.init()

    screen = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
    FONT = pygame.font.Font("freesansbold.ttf", 18)
    pygame.display.set_caption("Drawing Digits")

    # screen.fill(0,0,0)

    running = True
    drawing = False

    number_xcord = []
    number_ycord = []
    image_cnt = 0
    PREDICT = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == MOUSEBUTTONDOWN:
                drawing = True

            if event.type == MOUSEBUTTONUP:
                drawing = False
                if (len(number_xcord) >= 1 and len(number_ycord) >= 1):
                    number_xcord = sorted(number_xcord)
                    number_ycord = sorted((number_ycord))

                    rect_min_x, rect_max_x = max(number_xcord[0] - boundary, 0), min(WINDOWSIZEX, number_xcord[-1] + boundary)
                    rect_min_y, rect_max_y = max(0, number_ycord[0] - boundary), min(number_ycord[-1] + boundary, WINDOWSIZEX)

                    number_xcord = []
                    number_ycord = []
                    img_arr = np.array(pygame.PixelArray(screen))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(
                        np.float64)

                    if IMAGESAVE:
                        cv2.imwrite("image.png")
                        image_cnt += 1

                    if PREDICT:
                        image = cv2.resize(img_arr, (28, 28))
                        image = np.pad(image, (10, 10), 'constant', constant_values=0)
                        image = cv2.resize(image, (28, 28)) / 10000



                       # plt.imshow(image, cmap=matplotlib.cm.binary)
                       # plt.axis("off")
                       # plt.show()

                        label = str(Labels[np.argmax(NN.drawTest(image))])
                        textSurface = FONT.render(label, True, RED, WHITE)
                        textRecObj = textSurface.get_rect()
                        textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                        screen.blit(textSurface, textRecObj)

            if event.type == MOUSEMOTION and drawing:
                x, y = event.pos
                pygame.draw.circle(screen, WHITE, (x, y), 6, 0)

                number_xcord.append(x)
                number_xcord.append(x+1);
                number_xcord.append(x - 1);
                number_ycord.append(y)
                number_ycord.append(y + 1);
                number_ycord.append(y - 1);
            if event.type == KEYDOWN:
                if event.unicode == "b":
                    screen.fill((BLACK))

        pygame.display.flip()



# Start up the neural network
NN = NeuralNetwork.NeuralNet()
NN.load()
# Menu and menu options
menu()
option = int(input("Enter your option: "))
while option != 0:
    if option == 1:
        NN.train()
    elif option ==2:
        NN.save()
    elif option == 3:
        NN.load()
    elif option == 4:
        app()
    else:
        print("Invalid Option")

    menu()
    option = int(input("Enter your option: "))

