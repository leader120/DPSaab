import os
import time
for i in range(1):
 t0=time.time()
 os.system("python ./Getkernel.py")
 os.system("python ./Getfeature.py")
 os.system("python ./Getweight.py")
 os.system("python ./test.py")
 print("------------------- End: time -> using %10f seconds" % (time.time() - t0))