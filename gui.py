import tkinter as tk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd, ttk
from PIL import ImageTk, Image as PILImage
from ops import ImageWrapper


class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pytoshop")
        self.root.geometry("1200x800")

        # Create main frame to organize layout
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Image display area
        self.image_label = tk.Label(self.main_frame)
        self.image_label.pack(side=tk.TOP, expand=True)

        # Initialize webcam capture
        self.cam = None
        self.base_image = None

        # Variables for user inputs
        self.new_width = tk.StringVar()
        self.new_height = tk.StringVar()
        self.rotation_degrees = tk.StringVar()
        self.vert_shear_value = tk.StringVar()
        self.horiz_shear_value = tk.StringVar()
        self.linear_a = tk.StringVar()
        self.linear_b = tk.StringVar()
        self.gamma_value = tk.StringVar()
        self.kernel_value = tk.StringVar()

        # For crop positions
        self.crop_start_x = tk.StringVar()
        self.crop_start_y = tk.StringVar()
        self.crop_end_x = tk.StringVar()
        self.crop_end_y = tk.StringVar()

        # Create frames for different operation groups
        self.create_basic_operations_frame()
        self.create_transformation_frame()
        self.create_more_transformation_frames()
        self.create_color_mapping_frame()
        self.create_filtering_frame()

        # Initialize image-related attributes
        self.base_image = None
        self.non_rotated_image = None
        self.image_wrapper = None
        self.curr_rotation = 0
        self.is_rotated = False

    def create_basic_operations_frame(self):
        """Create frame for basic image operations."""
        basic_frame = ttk.LabelFrame(self.main_frame, text="Basic Operations")
        basic_frame.pack(fill=tk.X, padx=10, pady=5)

        # Buttons for basic operations
        buttons = [
            ("Load Image", self.load_image),
            ("Flip Horizontal", self.flip_horizontal),
            ("Flip Vertical", self.flip_vertical),
            ("Open Image", self.open_image),
            ("Save Image", self.save_image),
            ("Capture from Webcam", self.open_webcam),
            ("Calculate Histogram", self.calculate_histogram),
            ("Equalize Histogram", self.equalize_histogram),
        ]

        for text, command in buttons:
            btn = ttk.Button(basic_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5)

    def create_transformation_frame(self):
        """Create frame for image transformations."""
        transform_frame = ttk.LabelFrame(self.main_frame, text="Transformations")
        transform_frame.pack(fill=tk.X, padx=10, pady=5)

        # Scaling
        ttk.Label(transform_frame, text="Scale (W x H):").pack(side=tk.LEFT)
        ttk.Entry(transform_frame, textvariable=self.new_width, width=5).pack(
            side=tk.LEFT
        )
        ttk.Entry(transform_frame, textvariable=self.new_height, width=5).pack(
            side=tk.LEFT
        )

        bilinear_var = tk.IntVar()
        ttk.Checkbutton(transform_frame, text="Bilinear", variable=bilinear_var).pack(
            side=tk.LEFT
        )

        ttk.Button(
            transform_frame,
            text="Scale",
            command=lambda: self.scale_image(
                self.new_width.get(), self.new_height.get(), bilinear_var.get()
            ),
        ).pack(side=tk.LEFT)

        # Rotation
        ttk.Label(transform_frame, text="Rotate (degrees):").pack(side=tk.LEFT)
        ttk.Entry(transform_frame, textvariable=self.rotation_degrees, width=5).pack(
            side=tk.LEFT
        )
        ttk.Button(
            transform_frame,
            text="Rotate",
            command=lambda: self.rotate_image(self.rotation_degrees.get(), 2),
        ).pack(side=tk.LEFT)

    def create_more_transformation_frames(self):
        more_transform_frame = ttk.LabelFrame(
            self.main_frame, text="More Transformations"
        )
        more_transform_frame.pack(fill=tk.X, padx=10, pady=5)

        # Crop Section
        crop_frame = ttk.LabelFrame(more_transform_frame, text="Crop")
        crop_frame.pack(fill=tk.X, padx=10, pady=5)

        # Crop Coordinate Inputs
        ttk.Label(crop_frame, text="Start X:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(crop_frame, textvariable=self.crop_start_x, width=5).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Label(crop_frame, text="Start Y:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(crop_frame, textvariable=self.crop_start_y, width=5).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Label(crop_frame, text="End X:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(crop_frame, textvariable=self.crop_end_x, width=5).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Label(crop_frame, text="End Y:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(crop_frame, textvariable=self.crop_end_y, width=5).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Button(
            crop_frame,
            text="Crop",
            command=lambda: self.crop_image(
                self.crop_start_x.get(),
                self.crop_start_y.get(),
                self.crop_end_x.get(),
                self.crop_end_y.get(),
            ),
        ).pack(side=tk.LEFT, padx=5)

        # Shear Section
        ttk.Label(more_transform_frame, text="Vertical Shear (0-1):").pack(side=tk.LEFT)
        ttk.Entry(
            more_transform_frame, textvariable=self.vert_shear_value, width=5
        ).pack(side=tk.LEFT)
        ttk.Button(
            more_transform_frame,
            text="Vertical Shear",
            command=lambda: self.vertical_shear(self.vert_shear_value.get()),
        ).pack(side=tk.LEFT)
        ttk.Label(more_transform_frame, text="Horizontal Shear (0-1):").pack(
            side=tk.LEFT
        )
        ttk.Entry(
            more_transform_frame, textvariable=self.horiz_shear_value, width=5
        ).pack(side=tk.LEFT)
        ttk.Button(
            more_transform_frame,
            text="Horizontal Shear",
            command=lambda: self.horizontal_shear(self.horiz_shear_value.get()),
        ).pack(side=tk.LEFT)

    def create_color_mapping_frame(self):
        """Create frame for color mapping operations."""
        mapping_frame = ttk.LabelFrame(self.main_frame, text="Color Mapping")
        mapping_frame.pack(fill=tk.X, padx=10, pady=5)

        # Linear mapping
        ttk.Label(mapping_frame, text="Linear Mapping (a, b):").pack(side=tk.LEFT)
        ttk.Entry(mapping_frame, textvariable=self.linear_a, width=5).pack(side=tk.LEFT)
        ttk.Entry(mapping_frame, textvariable=self.linear_b, width=5).pack(side=tk.LEFT)
        ttk.Button(
            mapping_frame,
            text="Apply Linear",
            command=lambda: self.linear_mapping(
                self.linear_a.get(), self.linear_b.get()
            ),
        ).pack(side=tk.LEFT)

        # Power law mapping
        ttk.Label(mapping_frame, text="Gamma:").pack(side=tk.LEFT)
        ttk.Entry(mapping_frame, textvariable=self.gamma_value, width=5).pack(
            side=tk.LEFT
        )
        ttk.Button(
            mapping_frame,
            text="Power Law",
            command=lambda: self.power_mapping(self.gamma_value.get()),
        ).pack(side=tk.LEFT)

    def create_filtering_frame(self):
        """Create frame for image filtering operations."""
        filter_frame = ttk.LabelFrame(self.main_frame, text="Filters")
        filter_frame.pack(fill=tk.X, padx=10, pady=5)

        # Filtering buttons
        filter_buttons = [
            ("Edge Detection", self.edge_detection),
            ("Min Filter", self.min_filter),
            ("Median Filter", self.median_filter),
            ("Max Filter", self.max_filter),
        ]

        for text, command in filter_buttons:
            ttk.Button(filter_frame, text=text, command=command).pack(
                side=tk.LEFT, padx=5
            )

        # Convolution
        ttk.Label(filter_frame, text="Kernel: (ex. [[1,1,1],[1,1,1],[1,1,1]])").pack(
            side=tk.LEFT
        )
        ttk.Entry(filter_frame, textvariable=self.kernel_value, width=20).pack(
            side=tk.LEFT
        )
        ttk.Button(
            filter_frame,
            text="Convolution",
            command=lambda: self.convolution(self.kernel_value.get()),
        ).pack(side=tk.LEFT)

    def load_image(self):
        file_path = fd.askopenfilename(
            title="Select an Image", filetypes=[("Image Files", "*.*")]
        )
        if file_path:
            # Convert image to RGB mode to ensure consistent format
            self.base_image = PILImage.open(file_path).convert("RGB")
            self.non_rotated_image = self.base_image
            self.display_image(self.base_image)
            print(f"IMAGE DIMENSIONS {self.base_image.width}x{self.base_image.height}")

            # Convert PIL image to numpy array
            img_array = np.array(self.base_image)

            # Create ImageWrapper with correct dimensions
            self.image_wrapper = ImageWrapper(
                self.base_image.width, self.base_image.height
            )

            # Safely copy pixel data to C image structure
            try:
                for y in range(self.base_image.height):
                    for x in range(self.base_image.width):
                        pixel = img_array[y, x]
                        index = y * self.base_image.width + x
                        self.image_wrapper.img.contents.rgb[index].r = int(pixel[0])
                        self.image_wrapper.img.contents.rgb[index].g = int(pixel[1])
                        self.image_wrapper.img.contents.rgb[index].b = int(pixel[2])
            except Exception as e:
                print(f"Error copying image data: {e}")
                print(f"Image shape: {img_array.shape}")
                # Optionally, reset image_wrapper
                self.image_wrapper = None

    def open_webcam(self):
        """Capture an image from the webcam and load it in the app."""
        if self.cam is None:
            self.cam = cv2.VideoCapture(0)

        if not self.cam.isOpened():
            print("Error: Could not open webcam.")
            return

        # Capture a single frame
        ret, frame = self.cam.read()
        if ret:
            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to a PIL image
            pil_image = PILImage.fromarray(frame_rgb)

            # Update the base_image with the new captured frame
            self.base_image = pil_image
            self.update_image_wrapper_from_pil(self.base_image)

            # Display the captured image
            self.display_image(pil_image)

    # this is used if we do a python operation that manipulates image such as
    # equalize histogram then perform a C operation such as flipping
    def update_image_wrapper_from_pil(self, pil_image):
        # Convert PIL image to numpy array
        img_array = np.array(pil_image)

        # Update image_wrapper with new image data
        self.image_wrapper = ImageWrapper(pil_image.width, pil_image.height)
        for y in range(pil_image.height):
            for x in range(pil_image.width):
                pixel = img_array[y, x]
                index = y * pil_image.width + x
                if pil_image.mode == "L":  # Greyscale image
                    self.image_wrapper.img.contents.rgb[index].r = int(pixel)
                    self.image_wrapper.img.contents.rgb[index].g = int(pixel)
                    self.image_wrapper.img.contents.rgb[index].b = int(pixel)
                elif pil_image.mode == "RGB":  # RGB image
                    self.image_wrapper.img.contents.rgb[index].r = int(pixel[0])
                    self.image_wrapper.img.contents.rgb[index].g = int(pixel[1])
                    self.image_wrapper.img.contents.rgb[index].b = int(pixel[2])

    def display_image(self, pil_image):
        img_tk = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference to avoid garbage collection

    def open_image(self):
        if self.base_image:
            self.base_image.show()

    def save_image(self):
        if self.base_image:
            file_path = fd.asksaveasfilename(defaultextension=".png")
            if file_path:
                self.base_image.save(file_path)

    def scale_image(self, width, height, bilinear):
        if self.image_wrapper:
            self.image_wrapper.scale(int(width), int(height), int(bilinear))
            scaled_image = self.image_wrapper.to_pil_image()
            self.base_image = scaled_image
            self.display_image(self.base_image)
            self.non_rotated_image = self.base_image

    def rotate_image(self, degrees, n):
        if self.image_wrapper:
            if not self.is_rotated:
                self.is_rotated = True
                self.non_rotated_image = self.base_image
                self.curr_rotation = float(degrees)
            else:
                # Reset the image_wrapper with the original non-rotated image data
                self.base_image = self.non_rotated_image
                self.update_image_wrapper_from_pil(self.non_rotated_image)
                self.curr_rotation += float(degrees)

            self.image_wrapper.rotate(self.curr_rotation, n)
            rotated_image = self.image_wrapper.to_pil_image()
            self.base_image = rotated_image
            self.display_image(self.base_image)

    def crop_image(self, start_x, start_y, end_x, end_y):
        if self.base_image:
            image_width = self.base_image.width
            image_height = self.base_image.height

            # Perform bounds check
            if (
                int(start_x) < 0
                or int(start_y) < 0
                or int(end_x) > image_width
                or int(end_y) > image_height
                or int(start_x) >= int(end_x)
                or int(start_y) >= int(end_y)
            ):
                print("ERROR: Crop coordinates are out of bounds.")
                print(f"Image dimensions: {image_width}x{image_height}")
                print(f"Crop coordinates: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
                return  # Do not proceed with the crop if bounds are not valid

            if self.image_wrapper:
                self.image_wrapper.crop(
                    int(start_x), int(start_y), int(end_x), int(end_y)
                )
                cropped_image = self.image_wrapper.to_pil_image()
                self.base_image = cropped_image
                self.display_image(self.base_image)
                self.non_rotated_image = self.base_image

    def vertical_shear(self, offset):
        if float(offset) > 1 or float(offset) < 0:
            print("ERROR: Offset must be between 0 and 1")
            return
        if self.image_wrapper:
            self.image_wrapper.vertical_shear(float(offset))
            v_sheared_image = self.image_wrapper.to_pil_image()
            self.base_image = v_sheared_image
            self.display_image(self.base_image)
            self.non_rotated_image = self.base_image

    def horizontal_shear(self, offset):
        if float(offset) > 1 or float(offset) < 0:
            print("ERROR: Offset must be between 0 and 1")
            return
        if self.image_wrapper:
            self.image_wrapper.horizontal_shear(float(offset))
            h_sheared_image = self.image_wrapper.to_pil_image()
            self.base_image = h_sheared_image
            self.display_image(self.base_image)
            self.non_rotated_image = self.base_image

    def linear_mapping(self, a, b):
        # if empty inputs, revert to safe default values
        a = 1 if a == "" else float(a)
        b = 0 if b == "" else float(b)
        if self.image_wrapper:
            self.image_wrapper.linear_mapping(a, b)
            lin_image = self.image_wrapper.to_pil_image()
            self.base_image = lin_image
            self.display_image(self.base_image)
            self.non_rotated_image = self.base_image

    def power_mapping(self, gamma):
        # if empty, revert to safe default value
        gamma = 1 if gamma == "" else float(gamma)
        if self.image_wrapper:
            self.image_wrapper.power_mapping(gamma)
            power_image = self.image_wrapper.to_pil_image()
            self.base_image = power_image
            self.display_image(self.base_image)
            self.non_rotated_image = self.base_image

    def min_filter(self):
        if self.image_wrapper:
            self.image_wrapper.min_filter()
            min_image = self.image_wrapper.to_pil_image()
            self.base_image = min_image
            self.display_image(self.base_image)
            self.non_rotated_image = self.base_image

    def median_filter(self):
        if self.image_wrapper:
            self.image_wrapper.median_filter()
            median_image = self.image_wrapper.to_pil_image()
            self.base_image = median_image
            self.display_image(self.base_image)
            self.non_rotated_image = self.base_image

    def max_filter(self):
        if self.image_wrapper:
            self.image_wrapper.max_filter()
            max_image = self.image_wrapper.to_pil_image()
            self.base_image = max_image
            self.display_image(self.base_image)
            self.non_rotated_image = self.base_image

    def convolution(self, kernel):
        # Strip the outer square brackets
        kernel_string = kernel.strip()[1:-1]  # Remove the outermost [ and ]
        # Split the string into rows based on the '],[' separator
        rows = kernel_string.split("],[")
        # Initialize the list to store the kernel values
        k = []
        for row in rows:
            # Remove spaces and split by commas, ensuring that each value is clean
            clean_row = [x.strip(" []") for x in row.split(",")]
            # Convert each value in the row to an integer
            k.append([int(x) for x in clean_row])

        if self.image_wrapper:
            self.image_wrapper.convolution(k, len(k), len(k[0]))
            conv_image = self.image_wrapper.to_pil_image()
            self.base_image = conv_image
            self.display_image(self.base_image)
            self.non_rotated_image = self.base_image

    def flip_horizontal(self):
        if self.image_wrapper:
            self.image_wrapper.horizontal_flip()
            flipped_image = self.image_wrapper.to_pil_image()
            self.display_image(flipped_image)
            self.base_image = flipped_image
            self.non_rotated_image = self.base_image

    def flip_vertical(self):
        if self.image_wrapper:
            self.image_wrapper.vertical_flip()
            flipped_image = self.image_wrapper.to_pil_image()
            self.display_image(flipped_image)
            self.base_image = flipped_image
            self.non_rotated_image = self.base_image

    def edge_detection(self):
        if self.image_wrapper:
            self.image_wrapper.edge_detection()
            modified_image = self.image_wrapper.to_pil_image()
            self.display_image(modified_image)
            self.base_image = modified_image
            self.non_rotated_image = self.base_image

    def calculate_histogram(self):
        # if greyscale, we will have a distribution of grey levels
        if self.base_image and self.base_image.mode == "L":
            imageArr = np.array(self.base_image)
            imageArr = imageArr.astype("float32")

            _, _ = np.histogram(imageArr, bins=256)

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
            if self.base_image:
                r, g, b = self.base_image.split()
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

    def equalize_histogram(self):
        # To equalize, we must first generate a normalized histogram
        # then generate cumulative normalized histogram
        # multiply values by 255
        # done
        if self.base_image and self.base_image.mode == "L":
            # convert image to numpy array
            imageArr = np.array(self.base_image)

            # calculate normalized histogram
            hist, _ = np.histogram(imageArr, bins=256)
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
            newImage = PILImage.fromarray(equalizedPixels.astype("uint8"), mode="L")
            self.base_image = newImage
            self.non_rotated_image = self.base_image

            # create or update image_wrapper with the equalized image
            self.image_wrapper = ImageWrapper(
                self.base_image.width, self.base_image.height
            )
            self.update_image_wrapper_from_pil(self.base_image)
            self.display_image(self.base_image)

            # calculate normalized histogram for equalized image
            hist, _ = np.histogram(equalizedPixels, bins=256)
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
            imageArr = np.array(self.base_image)

            # calculate cumulative normalized new pixel values for each color channel
            equalizedPixels = np.zeros_like(imageArr)
            for channel in range(3):
                # calculate normalized histogram
                hist, _ = np.histogram(imageArr[:, :, channel], bins=256)
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
            newImage = PILImage.fromarray(equalizedPixels.astype("uint8"), mode="RGB")
            self.base_image = newImage
            self.non_rotated_image = self.base_image

            # create or update image_wrapper with the equalized image
            self.image_wrapper = ImageWrapper(
                self.base_image.width, self.base_image.height
            )
            self.update_image_wrapper_from_pil(self.base_image)
            self.display_image(self.base_image)

            # configure and draw the histogram figure
            plt.figure()
            plt.title("Normalized Cumulative RGB Histogram")
            plt.xlabel("RGB Value")
            plt.ylabel("Pixel Count")
            plt.xlim([0, 255])

            # plot normalized cumulative histograms for each color channel
            for channel, color in zip(range(3), ("r", "g", "b")):
                hist, _ = np.histogram(imageArr[:, :, channel], bins=256)
                normHist = hist / np.sum(hist)
                cumNormHist = np.cumsum(normHist)
                plt.plot(cumNormHist, color=color, alpha=0.5, label=color.capitalize())
            plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
