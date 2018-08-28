from matplotlib import pyplot as plt

def plotData(mySampleDataX, mySampleDataY, transferfunction, dpi, hertz):
    fig, ax = plt.subplots()
    ax.scatter(*zip(*mySampleDataX),color="b")
    plt.suptitle("Transferfunction: "+ transferfunction)
    plt.title("DPI= "+str(dpi) + "Hertz= "+str(hertz))
    plt.xlabel("Raw Mouse dx")
    plt.ylabel("dx after applying TF")
    #ax.scatter(*zip(*mySampleDataY),color="r")
    transferfunction = transferfunction.replace(":", "")
    plt.savefig("thesis\\plots\\plot_"+str(transferfunction)+"_DPI="+dpi+"_Hertz="+hertz+".png")