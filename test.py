import numpy
from PIL import Image
from numpy import array



def createMatrix(arr , width , height):
    a_list = []
    while(height+16 <=512):
        for i in range(width,width+16):
            a_list.append(arr[width:width+16, height:height+16])
            if(width+16 <= 512):
                width+=16
            else:
                width = 0
                height+=16
    return a_list


img = Image.open("Photo of Lena in ppm.jpg")
arr = array((img))
arr2 = createMatrix(arr , 16 , 16)
print len(arr2)
print arr2[0][0]