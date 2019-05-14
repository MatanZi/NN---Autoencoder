import numpy
from PIL import Image
from numpy import array


class ImageProduce:

    def __init__(self, path, slice_size):
        self.slice_size = slice_size
        self.path = path
        self.width, self.height, self.matrix = self.build_slicer()

    def build_slicer(self):
        img = Image.open(self.path)
        width, height = img.size
        pixels256 = self.split_to_matrix(list(img.getdata()), width, height)
        return width, height, pixels256

    def split_to_matrix(self, a_list, pic_width, pic_height):
        return array(numpy.split(array(list(map(self.divide256, a_list))), (pic_width*pic_height)/(self.slice_size*self.slice_size)))

    def reconstruct(self, mat_pix):
        pixels512 = self.split_to_matrix_pic(list(mat_pix), self.width, self.height )
        img2 = Image.fromarray(array(pixels512))
        img2.show()

    @staticmethod
    def divide256(n):
        return float(n) / 256

    @staticmethod
    def multi256(n):
        return n * 256

    def split_to_matrix_pic(self, a_list, pic_height, pic_width):
        c_list = numpy.reshape(a_list, pic_height*pic_width)
        b_list = array(list(map(self.multi256, c_list)))
        return array(numpy.split(b_list, pic_width))
