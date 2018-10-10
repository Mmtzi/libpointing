import numpy as np
from matplotlib import pyplot as plt
from math import pow, sqrt, acos, log2

def calcRegressionLine(x, y):
    # number of observations/points
    print(x)
    print(y)
    x = np.array(x)
    y = np.array(y)
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x - n * m_y * m_x)
    SS_xx = np.sum(x * x - n * m_x * m_x)

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    plot_regression_line(x,y ,(b_0, b_1))

def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    print("Estimated coefficients:\nb_0 = {}  \
        \nb_1 = {}".format(b[0], b[1]))

    # function to show plot
    plt.show()

def calcScoreOfAction(actCursorPos, oldCursorPos, distance, olddistance, targetPosition, targetSize, time, button):
    old_beeline = targetPosition[0] -oldCursorPos[0] , targetPosition[1] -oldCursorPos[1]
    new_beeline = targetPosition[0] -actCursorPos[0] , targetPosition[1] -actCursorPos[1]
    movement = actCursorPos[0] - oldCursorPos[0] , actCursorPos[1]- oldCursorPos[1]
    sk_old_new = old_beeline[0]* new_beeline[0] + old_beeline[1]* new_beeline[1]
    sk_old_mvt = old_beeline[0]* movement[0] + old_beeline[1]* movement[1]
    length_old_beeline = sqrt(pow(old_beeline[0],2)+pow(old_beeline[1],2))
    length_new_beeline = sqrt(pow(new_beeline[0],2)+pow(new_beeline[1],2))
    length_movement = sqrt(pow(movement[0],2)+pow(movement[1],2))
    if length_movement > 0 and length_old_beeline > 0:
        angle_atStart = np.abs(np.rad2deg(acos(min(abs(sk_old_mvt) / (length_old_beeline * length_movement),1))))
    else:
        angle_atStart = 0
    if length_old_beeline > 0 and length_new_beeline >0:
        angle_atTarget =np.abs(np.rad2deg(acos(min(abs(sk_old_new) / (length_old_beeline*length_new_beeline),1))))
    else:
        angle_atTarget = 0
    distFkt = pow(log2(distance/100+1.01)+1,2)

    movement = length_old_beeline-length_new_beeline
    if (movement) <= 0:
        movementToTargetPenalty = abs(movement)
    else:
        movementToTargetPenalty = 1 / (movement)
    movementToTargetPenalty = (movementToTargetPenalty*targetSize)
    if button and distance <=targetSize:
        hit = 10
    else:
        hit = 0
    #print("angleAtStart: "+str(round(angle_atStart,2)) + "  angleAtTarget: "+str(round(angle_atTarget,2)) +"  distance: "+str(round(distance,2))
                #+ "  distFkt: "+str(round(distFkt,2)) +"  rx: "+str(round(actCursorPos[0]-oldCursorPos[0],2)) + "  ry: "+str(round(actCursorPos[1]- oldCursorPos[1],2))
                #+ "  movement2targetp: "+str(round(movementToTargetPenalty,2))+ "  size: "+str(targetSize)+ "  time: "+str(time))

    score = round(log2(100/(angle_atStart + angle_atTarget*10+1))+(10/(movementToTargetPenalty+1))*distFkt,2)+hit

    #print("anglescore:"+str(round(100/(angle_atStart+ angle_atTarget*10+1),2)) +" movementScore"+str(round(10/(movementToTargetPenalty+1),2)))

    #print(score)
    return score