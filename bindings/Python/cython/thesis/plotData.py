from matplotlib import pyplot as scatterDx
from matplotlib import pyplot as histoDx
from matplotlib import pyplot as scatterTF
import os

def plotData(mySampleDataX, mySampleDataY, transferfunction, dpi, hertz):
    scatterTF.scatter(*zip(*mySampleDataX),color="b")
    scatterTF.suptitle("Transferfunction: "+ transferfunction)
    scatterTF.title("DPI= "+str(dpi) + "Hertz= "+str(hertz))
    scatterTF.xlabel("Raw Mouse dx")
    scatterTF.ylabel("dx after applying TF")
    #ax.scatter(*zip(*mySampleDataY),color="r")
    transferfunction = transferfunction.replace(":", "")
    try:
        scatterTF.savefig("thesis\\plots\\plot_"+str(transferfunction)+"_DPI="+dpi+"_Hertz="+hertz+".png")
        print("saved plot as: plot_"+str(transferfunction)+"_DPI="+dpi+"_Hertz="+hertz+".png")
    except:
        print("couldnt save plot...")

def plotHistogramm(listofData, dpi, hertz):
    print("ploting histogramm...")
    histoDx.hist(listofData, bins=254)
    try:
        histoDx.savefig("thesis\\plots\\histo_DPI="+str(dpi)+"_Hertz="+str(hertz)+".png")
        print("saved histogramm as: histo_DPI="+str(dpi)+"_Hertz="+str(hertz)+".png")
    except:
        print("couldnt save plot...")

def plotResults(predictedDX, realDX, epochs):
    print("ploting results...")
    scatterDx.scatter(realDX, predictedDX, color="b")
    i = 1
    while os.path.exists('ml\\logs\\sim_adv_dist_dx_epochs_' + str(epochs) + '-Iterations_' + str(i) + '.png'):
        i += 1
    else:
        try:
            scatterDx.savefig('ml\\logs\\sim_adv_dist_dx_epochs_' + str(epochs) + '-Iterations_' + str(i) + '.png')
            print("saved plot as: sim_adv_dist_dx_epochs_" + str(epochs) + '-Iterations_' + str(i) + '.png')
        except:
            print("couldnt save plot...")