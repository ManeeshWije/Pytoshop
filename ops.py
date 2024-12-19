from PIL import ImageFile, Image as PILImage
import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL("./image_operations.so")

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Define structures used by the C code
class RGBPixel(ctypes.Structure):
    _fields_ = [("r", ctypes.c_ubyte), ("g", ctypes.c_ubyte), ("b", ctypes.c_ubyte)]


class CImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("rgb", ctypes.POINTER(RGBPixel)),
    ]


# Set up function prototypes
lib.create_image.argtypes = [ctypes.c_int, ctypes.c_int]
lib.create_image.restype = ctypes.POINTER(CImage)

lib.horizontal_flip.argtypes = [ctypes.POINTER(CImage)]
lib.horizontal_flip.restype = None

lib.vertical_flip.argtypes = [ctypes.POINTER(CImage)]
lib.vertical_flip.restype = None

lib.scale.argtypes = [ctypes.POINTER(CImage), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.scale.restype = None

lib.rotate_image.argtypes = [ctypes.POINTER(CImage), ctypes.c_double, ctypes.c_int]
lib.rotate_image.restype = None

lib.vertical_shear.argtypes = [ctypes.POINTER(CImage), ctypes.c_float]
lib.vertical_shear.restype = None

lib.horizontal_shear.argtypes = [ctypes.POINTER(CImage), ctypes.c_float]
lib.horizontal_shear.restype = None

lib.crop.argtypes = [
    ctypes.POINTER(CImage),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.crop.restype = None

lib.linear_mapping.argtypes = [ctypes.POINTER(CImage), ctypes.c_float, ctypes.c_float]
lib.linear_mapping.restype = None

lib.power_mapping.argtypes = [ctypes.POINTER(CImage), ctypes.c_float]
lib.power_mapping.restype = None

lib.min_filter.argtypes = [ctypes.POINTER(CImage)]
lib.min_filter.restype = None

lib.median_filter.argtypes = [ctypes.POINTER(CImage)]
lib.median_filter.restype = None

lib.max_filter.argtypes = [ctypes.POINTER(CImage)]
lib.max_filter.restype = None

lib.convolution.argtypes = [
    ctypes.POINTER(CImage),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]
lib.convolution.restype = None

lib.edge_detection.argtypes = [ctypes.POINTER(CImage)]
lib.edge_detection.restype = None


# Wrapper functions for the C library
class ImageWrapper:
    def __init__(self, width, height):
        self.img = lib.create_image(width, height)
        self.width = width
        self.height = height

    def horizontal_flip(self):
        lib.horizontal_flip(self.img)

    def vertical_flip(self):
        lib.vertical_flip(self.img)

    def scale(self, new_width, new_height, is_bilinear):
        lib.scale(self.img, new_width, new_height, is_bilinear)

    def rotate(self, degrees, n):
        lib.rotate_image(self.img, degrees, n)

    def edge_detection(self):
        lib.edge_detection(self.img)

    def vertical_shear(self, offset):
        lib.vertical_shear(self.img, offset)

    def horizontal_shear(self, offset):
        lib.horizontal_shear(self.img, offset)

    def crop(self, start_x, start_y, end_x, end_y, mode):
        lib.crop(self.img, start_x, start_y, end_x, end_y, mode)

    def linear_mapping(self, a, b):
        lib.linear_mapping(self.img, a, b)

    def power_mapping(self, gamma):
        lib.power_mapping(self.img, gamma)

    def min_filter(self):
        lib.min_filter(self.img)

    def median_filter(self):
        lib.median_filter(self.img)

    def max_filter(self):
        lib.max_filter(self.img)

    def convolution(self, kernel, n, m):
        # Flatten the 2D list to 1D
        flattened_kernel = [item for sublist in kernel for item in sublist]
        # Create a ctypes array for the flattened kernel
        kernel_array = (ctypes.c_int * len(flattened_kernel))(*flattened_kernel)
        lib.convolution(self.img, kernel_array, n, m)

    def to_pil_image(self):
        """Convert the image data to a PIL Image."""
        width = self.img.contents.width
        height = self.img.contents.height
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                index = y * width + x
                img_array[y, x] = [
                    self.img.contents.rgb[index].r,
                    self.img.contents.rgb[index].g,
                    self.img.contents.rgb[index].b,
                ]

        return PILImage.fromarray(img_array)
