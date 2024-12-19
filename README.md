# Pytoshop

### Compiling and Running (only tested on Linux)

1.  Ensure you have Python installed
2.  Ensure you have a C compiler installed
3.  Create a Python virtual environment by running `python -m venv venv`
4.  Activate the venv by running `source venv/bin/activate`
5.  Install dependencies by running `pip install -r requirements.txt`
6.  Create the shared C library by running `make`
7.  Run `make run`

### What and Why?

- An easy to use Image Processing Toolbox that contains various operations to do on RGB and greyscale images
  - All image operations use C functions to speed up computations
- Built to get a better understanding of certain image processing techniques and how they work under the hood
- Also wanted to learn more about C -> Python interoperability using `ctypes`
- There are many more operations that can be added such as object detection, image labeling, and also some bugs to iron out
- Feel free to open a PR for literally anything
