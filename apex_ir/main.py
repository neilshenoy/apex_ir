import time, apex_ir, apex_ir2

from mss.windows import MSS as mss
from numba import jit
import numpy as np
import mss.tools
import cv2 as cv
import time



def main():

    apex = apex_ir2.Apex_IR()
    apex.run()






    #start = time.perf_counter()
    #finish = time.perf_counter()
    #print(f'Process took {round(finish - start, 4)} seconds.')


if __name__ == '__main__':
    main()
