# -*- coding: utf-8 -*-

from pylibpointing import PointingDevice, DisplayDevice, TransferFunction
from pylibpointing import PointingDeviceManager, PointingDeviceDescriptor
from bindings.Python.cython.thesis import plotData
import time

import sys

def cb_man(desc, wasAdded):
	print(desc)
	if wasAdded:
		print("was added")
	else:
		print("was removed")

pm = PointingDeviceManager()
PointingDevice.idle(100)
pm.addDeviceUpdateCallback(cb_man)

for desc in pm:
	print(desc)

"""
for desc in pm:
	print desc.devURI
	print desc.vendor, desc.product
	pdev = PointingDevice(desc.uri)
"""

pdev = PointingDevice.create("any:")
ddev = DisplayDevice.create("any:")

#naive:?cdgain=1
# Naive constant CD gain, i.e. pixels = gain * mickey
#constant:?cdgain=4&nosubpix=false
# Resolution-aware constant CD gain
#sigmoid:?gmin=1&v1=0.05&v2=0.2&gmax=6&nosubpix=false
# gmin is minimum gain applied when velocity <= v1, gmax is maximum gain when velocity >= v2.
# The gain is interpolated between gmin and gmax, if the input velocity is between v1 and v2
# does not change mousepointer
#system:?slider=3&epp=true
# slider range -5 bis 5 changes mousepointer
#interp:tf/darwin-14?f=f2&normalize=false
# created transferfunction
#interp:thesis?f=myTF&normalize=false
transferfunction = "system:"
shortName="systemMouseAcc"
hertz = str(pdev.getUpdateFrequency())
dpi = str(pdev.getResolution())
#print(pdev.getRawPointer().getResolution())
print(dpi)
print(hertz)
tfct = TransferFunction.create(transferfunction, pdev, ddev)

def cb_fct(timestamp, dx, dy, button):
    timestamp = (timestamp / 1000000000 - start)
    rx, ry = tfct.applyd(dx, dy, timestamp)
    print("%s: %d %d %d -> %.2f %.2f"%(str(timestamp), dx, dy, button, rx, ry ))
    mySampleDataX.append((dx, rx))
    mySampleDataY.append((dy, ry))
    sys.stdout.flush()
    if len(mySampleDataX) == 1000:
        print("enough Samples")
        print("Samples per Second: "+str(1000/timestamp))
        plotData.plotData(mySampleDataX, mySampleDataY, shortName, dpi, hertz)
        sys.exit()


mySampleDataX = []
mySampleDataY = []

start = time.time()

pdev.setCallback(cb_fct)

print("Move the mouse of Press CTRL+C to exit")

for i in range(0, 100000):
    PointingDevice.idle(1)
