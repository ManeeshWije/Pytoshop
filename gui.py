from tkinter import filedialog as fd, ttk
from Toolbox import Toolbox
from PIL import Image, ImageTk
import tkinter as tk
import cv2


root = tk.Tk()
root.title("Image Toolbox")
root.geometry("1000x1000")
baseFont = ("times", 18, "bold")


# variables that get passed in by user input
newWidth = tk.StringVar()
newHeight = tk.StringVar()
currentWidth = tk.StringVar()
currentHeight = tk.StringVar()
currentDimensions = tk.StringVar()
degrees = tk.StringVar()
xCrop = tk.StringVar()
yCrop = tk.StringVar()
hCrop = tk.StringVar()
wCrop = tk.StringVar()
shear = tk.StringVar()
a = tk.StringVar()
b = tk.StringVar()
gamma = tk.StringVar()
topX, topY, botX, botY = 0, 0, 0, 0
rectangleID = None
kernel = tk.StringVar()

canvas = tk.Canvas(
    root,
    width=1000,
    height=1000,
    borderwidth=0,
    highlightthickness=0,
)
label2 = ttk.Label(root)
label2.pack(side=tk.BOTTOM)

# for interactive cropping region
rectangleID = canvas.create_rectangle(0, 0, 0, 0, outline="white")


# record first click pos
def getMousePosition(event):
    global topX, topY
    topX, topY = int(canvas.canvasx(event.x)), int(canvas.canvasy(event.y))


# on mouse move, updated the coordinates then we know the region we should crop
def updateSelectionRect(event):
    global rectangleID
    global botX, botY
    botX, botY = int(canvas.canvasx(event.x)), int(canvas.canvasy(event.y))
    canvas.coords(rectangleID, topX, topY, botX, botY)
    canvas.lift(rectangleID)
    # print(canvas.coords(rectangleID))


def uploadFile():
    filetypes = [("all files", "*.*")]
    filename = fd.askopenfilename(filetypes=filetypes)

    img = Image.open(filename)

    global toolboxImage
    toolboxImage = Toolbox(img)  # create a new class after uploading

    tkImage = ImageTk.PhotoImage(img)
    canvas.config(width=img.width, height=img.height)
    canvas.img = tkImage  # Keep reference in case this code is put into a function.
    canvas.pack(expand=True)
    canvas.create_image(0, 0, image=tkImage, anchor=tk.NW)

    # always show updated dimensions
    currentWidth = str(img.width)
    currentHeight = str(img.height)
    currentDimensions = currentWidth + "x" + currentHeight
    label2.configure(text=currentDimensions)

    return toolboxImage


# upon every operation, update image will be called to update the base image instance variable and the new width and height if necessary
def updateImage():
    tkImage = ImageTk.PhotoImage(toolboxImage.baseImage)
    canvas.config(
        width=toolboxImage.baseImage.width, height=toolboxImage.baseImage.height
    )
    canvas.img = tkImage  # Keep reference in case this code is put into a function.
    canvas.pack(expand=True)
    canvas.create_image(0, 0, image=tkImage, anchor=tk.NW)

    currentWidth = str(toolboxImage.baseImage.width)
    currentHeight = str(toolboxImage.baseImage.height)
    currentDimensions = currentWidth + "x" + currentHeight
    label2.configure(text=currentDimensions)


# each frame corresponds to a horizontal section of buttons and/or text input
buttonsFrame = ttk.Frame(root)
buttonsFrame.pack(side=tk.TOP, fill=tk.X)
buttonsFrame1 = ttk.Frame(root)
buttonsFrame1.pack(side=tk.TOP, fill=tk.X)
buttonsFrame2 = ttk.Frame(root)
buttonsFrame2.pack(side=tk.TOP, fill=tk.X)
buttonsFrame3 = ttk.Frame(root)
buttonsFrame3.pack(side=tk.TOP, fill=tk.X)
buttonsFrame4 = ttk.Frame(root)
buttonsFrame4.pack(side=tk.TOP, fill=tk.X)
buttonsFrame5 = ttk.Frame(root)
buttonsFrame5.pack(side=tk.TOP, fill=tk.X)
buttonsFrame6 = ttk.Frame(root)
buttonsFrame6.pack(side=tk.TOP, fill=tk.X)
buttonsFrame7 = ttk.Frame(root)
buttonsFrame7.pack(side=tk.TOP, fill=tk.X)
buttonsFrame8 = ttk.Frame(root)
buttonsFrame8.pack(side=tk.TOP, fill=tk.X)
buttonsFrame9 = ttk.Frame(root)
buttonsFrame9.pack(side=tk.TOP, fill=tk.X)

b1 = ttk.Button(buttonsFrame, text="Upload Image", command=uploadFile)
b1.pack(side=tk.LEFT)

b2 = ttk.Button(
    buttonsFrame, text="Open Image", command=lambda: [toolboxImage.baseImage.show()]
)
b2.pack(side=tk.LEFT)

b3 = ttk.Button(
    buttonsFrame,
    text="Save Image",
    command=lambda: [toolboxImage.baseImage.save("toolboxImage.png")],
)
b3.pack(side=tk.LEFT)

b4 = ttk.Button(
    buttonsFrame1,
    text="Horizontally Flip",
    command=lambda: [toolboxImage.horizontalFlip(), updateImage()],
)
b4.pack(side=tk.LEFT)
b5 = ttk.Button(
    buttonsFrame1,
    text="Vertically Flip",
    command=lambda: [toolboxImage.verticalFlip(), updateImage()],
)
b5.pack(side=tk.LEFT)

bilinear = tk.IntVar()
# this value will be checked in the toolbox func to see if they want to perform bilinear interpolation
c3 = tk.Checkbutton(
    buttonsFrame2,
    text="Bilinear Interpolation",
    variable=bilinear,
    onvalue=1,
    offvalue=0,
)

l1 = ttk.Label(buttonsFrame2, text="Enter New Dimensions (width, height): ")
l1.pack(side=tk.LEFT)
c3.pack(side=tk.LEFT)

t1 = ttk.Entry(buttonsFrame2, textvariable=newWidth, width=5)
t1.pack(side=tk.LEFT)

t2 = ttk.Entry(buttonsFrame2, textvariable=newHeight, width=5)
t2.pack(side=tk.LEFT)

b6 = ttk.Button(
    buttonsFrame2,
    text="Scale Image",
    command=lambda: [
        toolboxImage.scale(newWidth.get(), newHeight.get(), bilinear.get()),
        updateImage(),
    ],
)

b6.pack(side=tk.LEFT)

l2 = ttk.Label(buttonsFrame3, text="Enter degrees of rotation: ")
l2.pack(side=tk.LEFT)

t3 = ttk.Entry(buttonsFrame3, textvariable=degrees, width=5)
t3.pack(side=tk.LEFT)
b7 = ttk.Button(
    buttonsFrame3,
    text="Rotate Image",
    command=lambda: [
        toolboxImage.rotate(degrees.get()),
        updateImage(),
    ],
)
b7.pack(side=tk.LEFT)

l3 = ttk.Label(
    buttonsFrame4, text="Use the mouse on the image to select area to crop: "
)
l3.pack(side=tk.LEFT)

# check if the user wants to perform circular indexing or reflected indexing
circCrop = tk.IntVar()
reflectCrop = tk.IntVar()
c2 = tk.Checkbutton(
    buttonsFrame4, text="Circular Indexing", variable=circCrop, onvalue=1, offvalue=0
)
c2.pack(side=tk.LEFT)
c2r = tk.Checkbutton(
    buttonsFrame4,
    text="Reflected Indexing",
    variable=reflectCrop,
    onvalue=1,
    offvalue=0,
)
c2r.pack(side=tk.LEFT)

# coords are coming from initial mouse click and drag to last position of mouse
b8 = ttk.Button(
    buttonsFrame4,
    text="Crop Image",
    command=lambda: [
        toolboxImage.crop(
            canvas.coords(rectangleID)[0],
            canvas.coords(rectangleID)[1],
            canvas.coords(rectangleID)[2],
            canvas.coords(rectangleID)[3],
            circCrop.get(),
            reflectCrop.get(),
        ),
        updateImage(),
    ],
)
b8.pack(side=tk.LEFT)

l4 = ttk.Label(buttonsFrame5, text="Enter offset for shear: ")
l4.pack(side=tk.LEFT)

t4 = ttk.Entry(buttonsFrame5, textvariable=shear, width=5)
t4.pack(side=tk.LEFT)

b9 = ttk.Button(
    buttonsFrame5,
    text="Horizontal Shear",
    command=lambda: [
        toolboxImage.horizontalShear(shear.get()),
        updateImage(),
    ],
)
b9.pack(side=tk.LEFT)

b10 = ttk.Button(
    buttonsFrame5,
    text="Vertical Shear",
    command=lambda: [
        toolboxImage.verticalShear(shear.get()),
        updateImage(),
    ],
)
b10.pack(side=tk.LEFT)

l5 = ttk.Label(buttonsFrame6, text="Enter values for linear mapping: (a, b)")
l5.pack(side=tk.LEFT)

t5 = ttk.Entry(buttonsFrame6, textvariable=a, width=5)
t5.pack(side=tk.LEFT)

t5b = ttk.Entry(buttonsFrame6, textvariable=b, width=5)
t5b.pack(side=tk.LEFT)

b11 = ttk.Button(
    buttonsFrame6,
    text="Apply",
    command=lambda: [
        toolboxImage.linearMapping(a.get(), b.get()),
        updateImage(),
    ],
)
b11.pack(side=tk.LEFT)


l6 = ttk.Label(buttonsFrame7, text="Enter gamma value for power law mapping: ")
l6.pack(side=tk.LEFT)

t6 = ttk.Entry(buttonsFrame7, textvariable=gamma, width=5)
t6.pack(side=tk.LEFT)

b12 = ttk.Button(
    buttonsFrame7,
    text="Apply",
    command=lambda: [
        toolboxImage.powerLawMapping(gamma.get()),
        updateImage(),
    ],
)
b12.pack(side=tk.LEFT)


# open live webcam window
def openWebcam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
    while True:
        ret, frame = cap.read()
        cv2.imshow("press <Space> to take a picture and 'q' to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif cv2.waitKey(1) & 0xFF == ord(" "):
            cv2.imwrite("webcamImage.jpg", frame)

            img = Image.open("webcamImage.jpg")
            global toolboxImage
            toolboxImage = Toolbox(img)
            tkImage = ImageTk.PhotoImage(img)
            canvas.config(
                width=toolboxImage.baseImage.width, height=toolboxImage.baseImage.height
            )
            canvas.img = tkImage
            canvas.pack(expand=True)
            canvas.create_image(0, 0, image=tkImage, anchor=tk.NW)

            currentWidth = str(toolboxImage.baseImage.width)
            currentHeight = str(toolboxImage.baseImage.height)
            currentDimensions = currentWidth + "x" + currentHeight
            label2.configure(text=currentDimensions)
            break
    cap.release()
    cv2.destroyAllWindows()


bw1 = ttk.Button(
    buttonsFrame,
    text="Use Webcam",
    command=lambda: [openWebcam()],
)
bw1.pack(side=tk.LEFT)

b13 = ttk.Button(
    buttonsFrame,
    text="Calculate Histogram",
    command=lambda: [
        toolboxImage.generateHistogram(),
        updateImage(),
    ],
)
b13.pack(side=tk.LEFT)

b14 = ttk.Button(
    buttonsFrame,
    text="Equalize Histogram",
    command=lambda: [
        toolboxImage.generateEqualizedHistogram(),
        updateImage(),
    ],
)
b14.pack(side=tk.LEFT)

l7 = ttk.Label(
    buttonsFrame8, text="Enter 2D array representing kernel (must include '[]')"
)
l7.pack(side=tk.LEFT)

t7 = ttk.Entry(buttonsFrame8, textvariable=kernel, width=25)
t7.pack(side=tk.LEFT)

b15 = ttk.Button(
    buttonsFrame8,
    text="Apply Convolution",
    command=lambda: [
        toolboxImage.convolution(kernel.get()),
        updateImage(),
    ],
)
b15.pack(side=tk.LEFT)

b16 = ttk.Button(
    buttonsFrame9,
    text="Min Filter",
    command=lambda: [
        toolboxImage.minFilter(),
        updateImage(),
    ],
)
b16.pack(side=tk.LEFT)

b17 = ttk.Button(
    buttonsFrame9,
    text="Median Filter",
    command=lambda: [
        toolboxImage.medianFilter(),
        updateImage(),
    ],
)
b17.pack(side=tk.LEFT)

b18 = ttk.Button(
    buttonsFrame9,
    text="Max Filter",
    command=lambda: [
        toolboxImage.maxFilter(),
        updateImage(),
    ],
)
b18.pack(side=tk.LEFT)

b19 = ttk.Button(
    buttonsFrame9,
    text="Edge Detection",
    command=lambda: [
        toolboxImage.edgeDetection(),
        updateImage(),
    ],
)
b19.pack(side=tk.LEFT)

canvas.bind("<Button-1>", getMousePosition)
canvas.bind("<B1-Motion>", updateSelectionRect)

root.mainloop()
