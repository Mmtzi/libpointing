import numpy as np
from matplotlib import pyplot as plt
from math import pow, sqrt, acos

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

def calcScoreOfAction(actCursorPos, oldCursorPos, distance, targetPosition, targetSize):
    old_dir = targetPosition[0] -oldCursorPos[0] , targetPosition[1] -oldCursorPos[1]
    new_dir = targetPosition[0] -actCursorPos[0] , targetPosition[1] -actCursorPos[1]
    act_dir = actCursorPos[0] - oldCursorPos[0] , actCursorPos[1]- oldCursorPos[1]
    skOldNew = old_dir[0]* new_dir[0] + old_dir[1]* new_dir[1]
    skOldAct = old_dir[0]* act_dir[0] + old_dir[1]* act_dir[1]
    length_old_dir = sqrt(pow(old_dir[0],2)+pow(old_dir[1],2))
    length_new_dir = sqrt(pow(new_dir[0],2)+pow(new_dir[1],2))
    length_act_dir = sqrt(pow(act_dir[0],2)+pow(act_dir[1],2))
    if length_act_dir > 0 and length_old_dir > 0:
        angleactold = acos(min(abs(skOldAct) / (length_old_dir * length_act_dir),1))
    else:
        angleactold = None
    if length_old_dir > 0 and length_new_dir >0:
        angleoldnew =acos(min(abs(skOldNew) / (length_old_dir*length_new_dir),1))
    else:
        angleoldnew = None
    print(angleactold, angleoldnew)

