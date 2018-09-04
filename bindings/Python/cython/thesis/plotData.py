from matplotlib import pyplot
import os

def plotData(mySampleDataX, mySampleDataY, transferfunction, dpi, hertz):
    tfScatter = pyplot
    tfScatter.scatter(*zip(*mySampleDataX),color="b")
    tfScatter.suptitle("Transferfunction: "+ transferfunction)
    tfScatter.title("DPI= "+str(dpi) + "Hertz= "+str(hertz))
    tfScatter.xlabel("Raw Mouse dx")
    tfScatter.ylabel("dx after applying TF")
    #ax.scatter(*zip(*mySampleDataY),color="r")
    transferfunction = transferfunction.replace(":", "")
    try:
        tfScatter.savefig("thesis\\plots\\plot_"+str(transferfunction)+"_DPI="+dpi+"_Hertz="+hertz+".png")
        print("saved plot as: plot_"+str(transferfunction)+"_DPI="+dpi+"_Hertz="+hertz+".png")
    except:
        print("couldnt save plot...")

def plotHistogramm(listofData, dpi, hertz):
    print("ploting histogramm...")
    dxHisto = pyplot.subplot()
    dxHisto.hist(listofData, bins=254)
    try:
        dxHisto.savefig("thesis\\plots\\histo_DPI="+str(dpi)+"_Hertz="+str(hertz)+".png")
        print("saved histogramm as: histo_DPI="+str(dpi)+"_Hertz="+str(hertz)+".png")
    except:
        print("couldnt save plot...")

def plotResults(predictedDX, realDX, epochs):
    print("ploting results...")
    predDxScatter = pyplot.subplot()
    predDxScatter.scatter(realDX, predictedDX, color="b")
    i = 1
    while os.path.exists('ml\\logs\\sim_adv_dist_dx_epochs_' + str(epochs) + '-Iterations_' + str(i) + '.png'):
        i += 1
    else:
        try:
            predDxScatter.savefig('ml\\logs\\sim_adv_dist_dx_epochs_' + str(epochs) + '-Iterations_' + str(i) + '.png')
            print("saved plot as: sim_adv_dist_dx_epochs_" + str(epochs) + '-Iterations_' + str(i) + '.png')
        except:
            print("couldnt save plot...")