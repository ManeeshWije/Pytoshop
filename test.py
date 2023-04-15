from PIL import Image

# Opens an image
bg = Image.open("./hank.png")

# The width and height of the background tile
bg_w, bg_h = bg.size

# Creates a new empty image, RGB mode, and size 1000 by 1000
new_im = Image.new("RGB", (1000, 1000))

# The width and height of the new image
w, h = new_im.size

# Iterate through a grid, to place the background tile
for i in range(0, w, bg_w):
    for j in range(0, h, bg_h):
        # Change brightness of the images, just to emphasise they are unique copies
        bg = Image.eval(bg, lambda x: x + (i + j) / 1000)

        # paste the image at location i, j:
        new_im.paste(bg, (i, j))

new_im.show()
