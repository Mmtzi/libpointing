
file = open("myTF.dat", "w")
file.write("#myfunction")
file.write("\n")
file.write("max-counts: 127\n")
file.write("\n")
file.write("# counts: pixels\n")
file.write("0: 0\n")
d=0.3
k=0.2
for i in range (1,128):
    file.write(str(i)+": "+ str("%.2f" %d)+"\n")
    d = k+i
    k= k+0.3
    i+=1
file.close()