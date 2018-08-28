# file: pointingdevice.pxd

from libcpp.string cimport string
from libcpp cimport bool   
from curi cimport URI   
   
cdef extern from "pointing/input/PointingDevice.h" namespace "pointing::PointingDevice":

    ctypedef long long int	 inttime "pointing::TimeStamp::inttime" 

    ctypedef void (*PointingCallback)(void *context, 
                                      int timestamp,
                                      int dx, int dy, 
                                      int buttons)

    cdef PointingDevice* create(char* my)

    cdef void idle(int milliseconds)

cdef extern from "pointing/input/PointingDevice.h" namespace "pointing":

    cdef cppclass PointingDevice:
        void setPointingCallback(PointingCallback callback, void *context)        
        double getResolution(double *defval)
        double getUpdateFrequency(double *defval)
        URI getURI(bool expanded)
        string getVendor()
        int getVendorID()
        string getProduct()
        int getProductID()