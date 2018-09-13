from matplotlib import pyplot as plt
import os

def plotData(mySampleDataX, mySampleDataY, transferfunction, dpi, hertz):
    plt.Figure()
    plt.scatter(*zip(*mySampleDataX),color="b")
    plt.suptitle("Transferfunction: "+ transferfunction)
    plt.title("DPI= "+str(dpi) + "Hertz= "+str(hertz))
    plt.xlabel("Raw Mouse dx")
    plt.ylabel("dx after applying TF")
    #ax.scatter(*zip(*mySampleDataY),color="r")
    transferfunction = transferfunction.replace(":", "")
    try:
        plt.savefig("thesis\\plots\\plot_"+str(transferfunction)+"_DPI="+dpi+"_Hertz="+hertz+".png")
        print("saved plot as: plot_"+str(transferfunction)+"_DPI="+dpi+"_Hertz="+hertz+".png")
    except:
        print("couldnt save plot...")

def plotHistogramm(listofData, dpi, hertz):
    print("ploting histogramm...")
    plt.Figure()
    plt.hist(listofData, bins=254)
    try:
        plt.savefig("thesis\\plots\\histo_DPI="+str(dpi)+"_Hertz="+str(hertz)+".png")
        print("saved histogramm as: histo_DPI="+str(dpi)+"_Hertz="+str(hertz)+".png")
    except:
        print("couldnt save plot...")

def plotResults(predictedDX, realDX, epochs):
    print("ploting results...")
    plt.Figure()
    plt.scatter(realDX, predictedDX, color="b")
    i = 1
    while os.path.exists('ml\\logs\\sim_adv_dist_dx_epochs_' + str(epochs) + '-Iterations_' + str(i) + '.png'):
        i += 1
    else:
        try:
            plt.savefig('ml\\logs\\sim_adv_dist_dx_epochs_' + str(epochs) + '-Iterations_' + str(i) + '.png')
            print("saved plot as: sim_adv_dist_dx_epochs_" + str(epochs) + '-Iterations_' + str(i) + '.png')
        except:
            print("couldnt save plot...")

def plotFitsDependencies(scorePlotList, dpi, hertz):
    print("ploting FitsDependencies...")
    plt.Figure()
    plt.scatter(*zip(*scorePlotList), color="r")
    plt.title("Fits Analysis: ")
    plt.xlabel("Level of Difficulty (ID)")
    plt.ylabel("Movement Time (MT) ")
    i = 1
    while os.path.exists("thesis\\plots\\fits\\fits_DPI="+str(dpi)+"_Hertz="+str(hertz)+"_"+str(i)+".png"):
        i += 1
    else:
        try:
            plt.savefig("thesis\\plots\\fits\\fits_DPI="+str(dpi)+"_Hertz="+str(hertz)+"_"+str(i)+".png")
            print("saved plot as: fits_DPI="+str(dpi)+"_Hertz="+str(hertz)+"_"+str(i)+".png")
        except:
            print("couldnt save plot...")


