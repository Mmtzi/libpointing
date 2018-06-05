# -*- coding: utf-8 -*-

from pylibpointing import PointingDevice, DisplayDevice, TransferFunction

import sys

try:
    pdev=PointingDevice.create(sys.argv[1])
except:
    pdev=PointingDevice.create("any:?debugLevel=1")
    
ddev=DisplayDevice.create("any:")
tfct=TransferFunction.create("system:", pdev, ddev)

print("========= Testing DisplayDevice =========")
#print("getURI: "+str(ddev.getURI()))
#print("getURI (w/ property): "+str(ddev.uri))
print("getBounds (in pixels): "+str(ddev.getBounds()))
print("getBounds (w/ property):"+str(ddev.bounds))
print("getSize (in mm): "+str(ddev.getSize()))
print("getSize (w/ property): "+str(ddev.size))
print("getResolution (in ppi): "+str(ddev.getResolution()))
print("getResolution (w/ property): "+str(ddev.resolution))
print("getRefreshRate (in Hz): "+str(ddev.getRefreshRate()))
print("getRefreshRate (w/ property): "+str(ddev.refreshrate)) 
    
print("========= Testing PointingDevice =========")
class ObjectCB(object):
    def __init__(self, name):
        self.name_=name
        self.done_=False
        
    def __call__(self, timestamp, dx, dy, buttons):
        if not buttons == 0:
            self.done_=True
        if not self.done_:
            print("objectCB: "+str(timestamp)+": "
                              +str(dx)+", "
                              +str(dy)+" "
                              +str(buttons)) 

    def done(self):
        return self.done_

#print("  getURI: "+str(pdev.getURI()))
#print("  getURI (w/ property): "+str(pdev.uri))
print("  getResolution (in ppi): "+str(pdev.getResolution()))
print("  getResolution (w/ property): "+str(pdev.resolution))
print("  frequency (in Hz): "+str(pdev.getUpdateFrequency()))
print("  frequency (w/ property): "+str(pdev.updatefrequency))

print("Move the PointingDevice to test or CTRL-C to exit")
ocb=ObjectCB("Roger")
pdev.setCallback(ocb)    
while not ocb.done():
    PointingDevice.idle(1000)

print("========= Testing Transfer Function =========")
def cb_fct(timestamp, dx, dy, button):
    rx,ry=tfct.apply(dx, dy, timestamp)
    print("%s: %d %d %d -> %d %d"%(str(timestamp), dx, dy, button, rx, ry ))

print("Move the mouse of Press CTRL+C to exit")
PointingDevice.idle(1000)
pdev.setCallback(cb_fct)
for i in range(0, 100):
    PointingDevice.idle(1000)
