from PIL import Image, ImageFile
import numpy as np
from matplotlib import pyplot as plt
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Toolbox:
    def __init__(self, baseImage):
        self.baseImage = baseImage
        self.currRotation = 0
        self.nonRotatedImage = None
        self.isRotated = False
        self.scaledWidth = 0
        self.scaledHeight = 0

    def getImageAttributes(self):
        return (
            self.baseImage,
            self.baseImage.width,
            self.baseImage.height,
            self.baseImage.load(),
        )

    def horizontalFlip(self):
        imgCopy, width, height, pixelMap = self.getImageAttributes()

        # loop through only half the width reassigning pixels from width - i indices
        for i in range(width // 2):
            for j in range(height):
                pixelMap[i, j], pixelMap[width - i - 1, j] = (
                    pixelMap[width - i - 1, j],
                    pixelMap[i, j],
                )
        self.baseImage = imgCopy

    def verticalFlip(self):
        imgCopy, width, height, pixelMap = self.getImageAttributes()

        # loop through only half of the height reassigning pixels from height - j indices
        for i in range(width):
            for j in range(height // 2):
                pixelMap[i, j], pixelMap[i, height - j - 1] = (
                    pixelMap[i, height - j - 1],
                    pixelMap[i, j],
                )
        self.baseImage = imgCopy

    def scale(self, newWidth, newHeight, option):
        imgCopy, width, height, pixelMap = self.getImageAttributes()

        # calculate ratios to use as scaling factors
        # this will be used for nearest neighbor scaling
        widthRatio = width / int(newWidth)
        heightRatio = height / int(newHeight)

        # construct the new image with the new user width and height
        if imgCopy.mode == "RGB":
            imgCopy = Image.new("RGB", (int(newWidth), int(newHeight)))
        else:
            imgCopy = Image.new("L", (int(newWidth), int(newHeight)))

        # loop through new dimensions and calculate updated positions for indices
        # then assign pixel values to updated positions
        for i in range(int(newWidth)):
            for j in range(int(newHeight)):
                # do bilinear interpolation
                if option == 1:
                    newI = i * (width - 1) / (int(newWidth) - 1)
                    newJ = j * (height - 1) / (int(newHeight) - 1)

                    i1, i2 = int(newI), int(newI) + 1
                    j1, j2 = int(newJ), int(newJ) + 1

                    alpha = newI - i1
                    beta = newJ - j1

                    if i2 >= width:
                        i2 = width - 1
                    if j2 >= height:
                        j2 = height - 1

                    if imgCopy.mode == "RGB":
                        r1 = (
                            pixelMap[i1, j1][0] * (1 - alpha)
                            + pixelMap[i2, j1][0] * alpha
                        )
                        g1 = (
                            pixelMap[i1, j1][1] * (1 - alpha)
                            + pixelMap[i2, j1][1] * alpha
                        )
                        b1 = (
                            pixelMap[i1, j1][2] * (1 - alpha)
                            + pixelMap[i2, j1][2] * alpha
                        )

                        r2 = (
                            pixelMap[i1, j2][0] * (1 - alpha)
                            + pixelMap[i2, j2][0] * alpha
                        )
                        g2 = (
                            pixelMap[i1, j2][1] * (1 - alpha)
                            + pixelMap[i2, j2][1] * alpha
                        )
                        b2 = (
                            pixelMap[i1, j2][2] * (1 - alpha)
                            + pixelMap[i2, j2][2] * alpha
                        )

                        r = r1 * (1 - beta) + r2 * beta
                        g = g1 * (1 - beta) + g2 * beta
                        b = b1 * (1 - beta) + b2 * beta

                        imgCopy.putpixel((i, j), (int(r), int(g), int(b)))
                    else:  # only exists one value for greyscale images
                        g1 = pixelMap[i1, j1] * (1 - alpha) + pixelMap[i2, j1] * alpha
                        g2 = pixelMap[i1, j2] * (1 - alpha) + pixelMap[i2, j2] * alpha
                        g = g1 * (1 - beta) + g2 * beta
                        imgCopy.putpixel((i, j), (int(g)))
                # do nearest neighbor
                else:
                    newI = i * widthRatio
                    newJ = j * heightRatio
                    imgCopy.putpixel((i, j), pixelMap[newI, newJ])
        self.baseImage = imgCopy
        # record this values so when combining operations, we use the new heights
        self.scaledWidth = self.baseImage.width
        self.scaledHeight = self.baseImage.height
        self.scaledPixelMap = self.baseImage.load()
        self.nonRotatedImage = self.baseImage

    def rotate(self, degrees, n=2):
        # this is so the bounding box resets after getting bigger and bigger
        if not self.isRotated:
            self.isRotated = True
            self.nonRotatedImage = self.baseImage
            self.currRotation += int(degrees)
        else:
            self.baseImage = self.nonRotatedImage
            self.currRotation += int(degrees)

        _, width, height, pixelMap = self.getImageAttributes()

        # First we will convert the degrees into radians
        rads = math.radians(self.currRotation)

        # Calculate the dimensions of the new image
        cosTheta = math.cos(rads)
        sinTheta = math.sin(rads)
        newWidth = int(abs(width * cosTheta) + abs(height * sinTheta))
        newHeight = int(abs(height * cosTheta) + abs(width * sinTheta))

        # construct the new image
        if self.baseImage.mode == "RGB":
            newImage = Image.new("RGB", (int(newWidth), int(newHeight)))
        else:
            newImage = Image.new("L", (int(newWidth), int(newHeight)))

        # calculate original origin and new origin
        center = (width // 2, height // 2)
        newCenter = (newWidth // 2, newHeight // 2)

        # oversample the source image to avoid aliasing
        # SUPER SLOW
        oversampledWidth = n * width
        oversampledHeight = n * height
        oversampledImage = Image.new(
            self.baseImage.mode, (oversampledWidth, oversampledHeight)
        )

        for i in range(width):
            for j in range(height):
                for k in range(n):
                    for l in range(n):
                        oversampledImage.putpixel(
                            (n * i + k, n * j + l), pixelMap[i, j]
                        )

        for i in range(oversampledWidth):
            for j in range(oversampledHeight):
                # calculate rotated coordinates and add the new origin to each
                x = i / n - center[0]
                y = j / n - center[1]
                newI = round(x * cosTheta - y * sinTheta) + newCenter[0]
                newJ = round(x * sinTheta + y * cosTheta) + newCenter[1]

                # since coord might go beyond the size of original, we consider pixels which are in bounds
                if newI >= 0 and newJ >= 0 and newI < newWidth and newJ < newHeight:
                    newImage.putpixel((newI, newJ), oversampledImage.getpixel((i, j)))
        self.baseImage = newImage

    def crop(self, x, y, w, h, circCrop, reflectCrop):
        imgCopy, width, height, pixelMap = self.getImageAttributes()

        if circCrop == 1 and reflectCrop == 0:
            # create new image and if in bounds, put those same pixels from original, otherwise do circular indexing
            if imgCopy.mode == "RGB":
                circIndexImage = Image.new("RGB", (width, height))
            else:
                circIndexImage = Image.new("L", (width, height))
            circIndexPixelMap = circIndexImage.load()
            tileSizeWidth = w - x
            tileSizeHeight = h - y
            for i in range(width):
                for j in range(height):
                    # if in region, we put the same pixels
                    if i > x and j > y and i < w and j < h:
                        circIndexImage.putpixel((i, j), pixelMap[i, j])
                    else:
                        # tile image on cropped region
                        circIndexPixelMap[i, j] = pixelMap[
                            (i - x) % (tileSizeWidth) + x,
                            (j - y) % (tileSizeHeight) + y,
                        ]
            self.baseImage = circIndexImage
            self.nonRotatedImage = self.baseImage
        elif circCrop == 0 and reflectCrop == 1:
            # create new image and if in bounds, put those same pixels from original, otherwise do circular indexing
            if imgCopy.mode == "RGB":
                reflectedIndexImage = Image.new("RGB", (width, height))
            else:
                reflectedIndexImage = Image.new("L", (width, height))
            reflectedIndexImagePixelMap = reflectedIndexImage.load()
            for i in range(width):
                for j in range(height):
                    # if in region, we put the same pixels
                    if i > x and j > y and i < w and j < h:
                        reflectedIndexImagePixelMap[i, j] = imgCopy.getpixel((i, j))
                    else:
                        # reflect based on the location we are in
                        if i < x:
                            newI = x - (i - x)
                        elif i >= w:
                            newI = w + (w - i) - 1
                        else:
                            newI = i
                        if j < y:
                            newJ = y - (j - y)
                        elif j >= h:
                            newJ = h + (h - j) - 1
                        else:
                            newJ = j
                        # ensure that the reflected index values are within bounds
                        newI = max(0, min(newI, width - 1))
                        newJ = max(0, min(newJ, height - 1))
                        reflectedIndexImagePixelMap[i, j] = imgCopy.getpixel(
                            (newI, newJ)
                        )
            self.baseImage = reflectedIndexImage
            self.nonRotatedImage = self.baseImage

        elif circCrop == 1 and reflectCrop == 1:
            print("ERROR: cannot select two options")
        # zero padding by default
        else:
            if imgCopy.mode == "RGB":
                newImage = Image.new("RGB", (width, height))
            else:
                newImage = Image.new("L", (width, height))
            newImagePixelMap = newImage.load()
            for i in range(width):
                for j in range(height):
                    if i > x and j > y and i < w and j < h:
                        newImagePixelMap[i, j] = pixelMap[i, j]
                    else:
                        newImagePixelMap[i, j] = 0
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage

    def horizontalShear(self, offset):
        imgCopy, width, height, pixelMap = self.getImageAttributes()
        offset = float(offset)

        # Calculate the maximum horizontal shift based on the shear factor
        maxShift = abs(int(height * offset))

        # horizontal shear will only affect the width
        newWidth = width + maxShift
        newHeight = height

        # construct the new image
        if imgCopy.mode == "RGB":
            imgCopy = Image.new("RGB", (int(newWidth), int(newHeight)))
        else:
            imgCopy = Image.new("L", (int(newWidth), int(newHeight)))

        for i in range(width):
            for j in range(height):
                # shear image width wise based on offset
                newI = i + int(j * offset)
                imgCopy.putpixel((newI, j), pixelMap[i, j])
        self.baseImage = imgCopy
        self.nonRotatedImage = self.baseImage

    def verticalShear(self, offset):
        imgCopy, width, height, pixelMap = self.getImageAttributes()
        offset = float(offset)

        # Calculate the maximum horizontal shift based on the shear factor
        maxShift = abs(int(width * offset))

        # vertical shear will only affect the height
        newWidth = width
        newHeight = height + maxShift

        # construct the new image
        if imgCopy.mode == "RGB":
            imgCopy = Image.new("RGB", (int(newWidth), int(newHeight)))
        else:
            imgCopy = Image.new("L", (int(newWidth), int(newHeight)))

        for i in range(width):
            for j in range(height):
                # shear image height wise based on offset
                newJ = j + int(i * offset)
                imgCopy.putpixel((i, newJ), pixelMap[i, j])
        self.baseImage = imgCopy
        self.nonRotatedImage = self.baseImage

    def linearMapping(self, a, b):
        imgCopy, width, height, _ = self.getImageAttributes()

        # if empty inputs, revert to safe default values
        a = 1 if a == "" else float(a)
        b = 0 if b == "" else float(b)

        if imgCopy.mode == "RGB":
            newImage = Image.new("RGB", (width, height))
            c = float(b)  # so we dont use same variable for greyScale
            for i in range(width):
                for j in range(height):
                    pixels = imgCopy.getpixel((i, j))
                    # apply linear equation to all 3 channels
                    r = int(a * pixels[0] + c)
                    g = int(a * pixels[1] + c)
                    b = int(a * pixels[2] + c)
                    newPixel = (r, g, b)
                    newImage.putpixel((i, j), newPixel)
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage
        else:
            newImage = Image.new("L", (width, height))
            for i in range(width):
                for j in range(height):
                    greyValue = imgCopy.getpixel((i, j))
                    # linear equation on each greyValue
                    newGreyValue = a * greyValue + b
                    newImage.putpixel((i, j), int(newGreyValue))
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage

    def powerLawMapping(self, gamma):
        imgCopy, width, height, _ = self.getImageAttributes()

        # if empty, revert to safe default value
        gamma = 1 if gamma == "" else float(gamma)

        if imgCopy.mode == "RGB":
            newImage = Image.new("RGB", (width, height))
            for i in range(width):
                for j in range(height):
                    pixels = imgCopy.getpixel((i, j))
                    # apply power-law equation to all 3 channels
                    r = int(255 * (pixels[0] / 255) ** gamma)
                    g = int(255 * (pixels[1] / 255) ** gamma)
                    b = int(255 * (pixels[2] / 255) ** gamma)
                    newPixel = (r, g, b)
                    newImage.putpixel((i, j), newPixel)
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage
        else:
            newImage = Image.new("L", (width, height))
            for i in range(width):
                for j in range(height):
                    greyValue = imgCopy.getpixel((i, j))
                    # power-law equation
                    newGreyValue = 255 * (greyValue / 255) ** gamma
                    newImage.putpixel((i, j), int(newGreyValue))
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage

    def generateHistogram(self):
        imgCopy, _, _, _ = self.getImageAttributes()

        # if greyscale, we will have a distribution of grey levels
        if imgCopy.mode == "L":
            imageArr = np.array(imgCopy)
            imageArr = imageArr.astype("float32")

            hist, bins = np.histogram(imageArr, bins=256)

            # configure and draw the histogram figure
            plt.figure()
            plt.title("Greyscale Histogram")
            plt.xlabel("Greyscale Value")
            plt.ylabel("Pixel Count")
            plt.xlim([0, 255])

            plt.hist(imageArr.flatten(), bins=256)
            plt.show()
        else:
            # if rgb, we will have a distribution of each of the 3 channels
            r, g, b = imgCopy.split()
            rArr, gArr, bArr = np.array(r), np.array(g), np.array(b)

            # configure and draw the histogram figure
            plt.figure()
            plt.title("RGB Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Pixel Count")
            plt.xlim([0, 255])

            plt.hist(rArr.flatten(), bins=256, color="r", alpha=0.5)
            plt.hist(gArr.flatten(), bins=256, color="g", alpha=0.5)
            plt.hist(bArr.flatten(), bins=256, color="b", alpha=0.5)
            plt.show()

    def generateEqualizedHistogram(self):
        imgCopy, _, _, _ = self.getImageAttributes()

        # To equalize, we must first generate a normalized histogram
        # then generate cumulative normalized histogram
        # multiply values by 255
        # done
        if imgCopy.mode == "L":
            # convert image to numpy array
            imageArr = np.array(imgCopy)

            # calculate normalized histogram
            hist, bins = np.histogram(imageArr, bins=256)
            normHist = hist / np.sum(hist)

            # calculate cumulative normalized new pixel values
            cdf = np.cumsum(normHist)
            cdfNormalized = 255 * cdf / cdf[-1]

            # find indices of pixels with intensity i
            equalizedPixels = np.zeros_like(imageArr)
            for i in range(256):
                idx = imageArr == i
                # set new intensity for those pixels
                equalizedPixels[idx] = cdfNormalized[i]

            # create new image based on the equalized pixels
            newImage = Image.fromarray(equalizedPixels.astype("uint8"), mode="L")
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage

            # calculate normalized histogram for equalized image
            hist, bins = np.histogram(equalizedPixels, bins=256)
            normHist = hist / np.sum(hist)

            # calculate normalized cumulative histogram for equalized image
            cumNormHist = np.cumsum(normHist)

            # configure and draw the histogram figure
            plt.figure()
            plt.title("Equalized Greyscale Histogram")
            plt.xlabel("Greyscale Value")
            plt.ylabel("Pixel Count")
            plt.xlim([0, 255])

            # plot normalized cumulative histogram for equalized image
            plt.plot(cumNormHist, color="k")
            plt.show()
        else:
            # convert image to numpy array
            imageArr = np.array(imgCopy)

            # calculate cumulative normalized new pixel values for each color channel
            equalizedPixels = np.zeros_like(imageArr)
            for channel in range(3):
                # calculate normalized histogram
                hist, bins = np.histogram(imageArr[:, :, channel], bins=256)
                normHist = hist / np.sum(hist)

                # calculate cumulative normalized new pixel values
                cdf = np.cumsum(normHist)
                cdfNormalized = 255 * cdf / cdf[-1]

                # find indices of pixels with intensity i
                for i in range(256):
                    idx = imageArr[:, :, channel] == i
                    # set new intensity for those pixels
                    equalizedPixels[idx, channel] = cdfNormalized[i]

            # create new image based on the equalized pixels
            newImage = Image.fromarray(equalizedPixels.astype("uint8"), mode="RGB")
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage
            imageArr = np.array(newImage)

            # configure and draw the histogram figure
            plt.figure()
            plt.title("Normalized Cumulative RGB Histogram")
            plt.xlabel("RGB Value")
            plt.ylabel("Pixel Count")
            plt.xlim([0, 255])

            # plot normalized cumulative histograms for each color channel
            for channel, color in zip(range(3), ("r", "g", "b")):
                hist, bins = np.histogram(imageArr[:, :, channel], bins=256)
                normHist = hist / np.sum(hist)
                cumNormHist = np.cumsum(normHist)
                plt.plot(cumNormHist, color=color, alpha=0.5, label=color.capitalize())
            plt.show()

    def convolution(self, kernel):
        # remove square brackets around outer array
        kernel = kernel.strip("[]")
        kernel = kernel.strip(" ")

        # split by '],[' to get rows
        rows = kernel.split("],[")

        # convert each string element to an integer
        kernel = [float(x) for row in rows for x in row.split(",")]

        # calculate the shape of the array
        kernelShape = (len(rows), len(rows[0].split(",")))

        # create a 2D array from the 1D list using numpy
        kernel = np.array(kernel).reshape(kernelShape)

        # make sure it is a valid convolution kernel
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            print("ERROR: cannot enter even kernel")
            return 1

        # apply convolution
        # [[-2,-1,0],[-1,1,1],[0,1,2]] to test
        imgCopy, width, height, _ = self.getImageAttributes()

        # calculate the padded input size
        padding = 20
        paddedWidth = width + padding * 2
        paddedHeight = height + padding * 2

        # create a padded copy of the input image
        paddedImage = Image.new("RGB", (paddedWidth, paddedHeight))
        paddedImage.paste(imgCopy, (padding, padding))

        # calculate the output size of the image
        newWidth = width - kernel.shape[1] + 1 + (2 * padding)
        newHeight = height - kernel.shape[0] + 1 + (2 * padding)

        newImage = Image.new("RGB", (paddedWidth, paddedHeight))

        # first we loop through the image itself then the kernel in order to apply it to each pixel
        for x in range(newWidth):
            for y in range(newHeight):
                pixelSum = [0, 0, 0]
                # iterate over each element in the kernel
                for i in range(kernel.shape[0]):
                    for j in range(kernel.shape[1]):
                        # get the pixel value at the corresponding location
                        currPixel = paddedImage.getpixel((x + i, y + j))
                        # multiply each pixel value by the kernel element and add to the sum
                        pixelSum[0] += int(currPixel[0] * kernel[i][j])
                        pixelSum[1] += int(currPixel[1] * kernel[i][j])
                        pixelSum[2] += int(currPixel[2] * kernel[i][j])
                newImage.putpixel((x, y), (pixelSum[0], pixelSum[1], pixelSum[2]))
        self.baseImage = newImage
        self.nonRotatedImage = self.baseImage

    def minFilter(self):
        imgCopy, width, height, _ = self.getImageAttributes()
        # if its greyscale, we only need to worry about 1 pixel value
        # otherwise, we need to gather r g and b channels
        if imgCopy.mode != "RGB":
            newImage = Image.new("L", (width, height))
            # truncate border when apply filter
            for i in range(1, width - 1):
                for j in range(1, height - 1):
                    pixels = []
                    # get 3x3 neighborhood pixels and append to pixels array
                    for x in range(i - 1, i + 2):
                        for y in range(j - 1, j + 2):
                            pixel = imgCopy.getpixel((x, y))
                            pixels.append(pixel)
                    # get the min and use that for the new pixel
                    newPixel = min(pixels)
                    newImage.putpixel((i, j), newPixel)
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage
        else:
            newImage = Image.new("RGB", (width, height))
            for i in range(1, width - 1):
                for j in range(1, height - 1):
                    r, g, b = [], [], []
                    for x in range(i - 1, i + 2):
                        for y in range(j - 1, j + 2):
                            # get all channels and get min of each to assign to new pixel
                            pixel = imgCopy.getpixel((x, y))
                            r.append(pixel[0])
                            g.append(pixel[1])
                            b.append(pixel[2])
                    newPixel = (min(r), min(g), min(b))
                    newImage.putpixel((i, j), newPixel)
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage

    def medianFilter(self):
        imgCopy, width, height, _ = self.getImageAttributes()
        if imgCopy.mode != "RGB":
            newImage = Image.new("L", (width, height))
            for i in range(1, width - 1):
                for j in range(1, height - 1):
                    pixels = []
                    for x in range(i - 1, i + 2):
                        for y in range(j - 1, j + 2):
                            pixel = imgCopy.getpixel((x, y))
                            pixels.append(pixel)
                    # for median, we need to sort and get middle value
                    # middle value is index 4 due to 3x3 neighborhood
                    pixels.sort()
                    newPixel = pixels[4]
                    newImage.putpixel((i, j), newPixel)
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage
        else:
            newImage = Image.new("RGB", (width, height))
            for i in range(1, width - 1):
                for j in range(1, height - 1):
                    r, g, b = [], [], []
                    for x in range(i - 1, i + 2):
                        for y in range(j - 1, j + 2):
                            pixel = imgCopy.getpixel((x, y))
                            r.append(pixel[0])
                            g.append(pixel[1])
                            b.append(pixel[2])
                    # sort all 3 channels seperately
                    r.sort()
                    g.sort()
                    b.sort()
                    # since sorted, we take the middle value of the 9 pixels for all 3 channels
                    newPixel = (r[4], g[4], b[4])
                    newImage.putpixel((i, j), newPixel)
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage

    def maxFilter(self):
        imgCopy, width, height, _ = self.getImageAttributes()
        if imgCopy.mode != "RGB":
            newImage = Image.new("L", (width, height))
            for i in range(1, width - 1):
                for j in range(1, height - 1):
                    pixels = []
                    for x in range(i - 1, i + 2):
                        for y in range(j - 1, j + 2):
                            pixel = imgCopy.getpixel((x, y))
                            pixels.append(pixel)
                    # take max value in pixels for new pixel value
                    newPixel = max(pixels)
                    newImage.putpixel((i, j), newPixel)
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage
        else:
            newImage = Image.new("RGB", (width, height))
            for i in range(1, width - 1):
                for j in range(1, height - 1):
                    r, g, b = [], [], []
                    for x in range(i - 1, i + 2):
                        for y in range(j - 1, j + 2):
                            pixel = imgCopy.getpixel((x, y))
                            r.append(pixel[0])
                            g.append(pixel[1])
                            b.append(pixel[2])
                    newPixel = (max(r), max(g), max(b))
                    newImage.putpixel((i, j), newPixel)
            self.baseImage = newImage
            self.nonRotatedImage = self.baseImage

    def edgeDetection(self):
        imgCopy, width, height, _ = self.getImageAttributes()
        # convert to grayscale before applying kernels
        imgCopy = imgCopy.convert("L")
        imgArr = np.array(imgCopy)

        # define the sobel operator kernels
        kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        newImageArr = np.zeros_like(imgArr)

        # ignore edge pixels and loop to height first because numpy
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # calculate the gradient magnitude using Sobel operator for x and y respectively
                gradientX = np.sum(kernelX * imgArr[i - 1 : i + 2, j - 1 : j + 2])
                gradientY = np.sum(kernelY * imgArr[i - 1 : i + 2, j - 1 : j + 2])
                magnitude = np.sqrt(gradientX**2 + gradientY**2)
                # set the output pixel value to the gradient mag
                newImageArr[i, j] = magnitude
        newImage = Image.fromarray(newImageArr.astype(np.uint8))
        # convert back so we can use existing transformation functions
        newImage = newImage.convert("RGB")
        self.baseImage = newImage
        self.nonRotatedImage = self.baseImage
