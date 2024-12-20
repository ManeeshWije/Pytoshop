#ifndef IMAGE_OPERATIONS_H
#define IMAGE_OPERATIONS_H

// Struct to represent a pixel in RGB mode
typedef struct {
    unsigned char r, g, b;
} RGBPixel;

// Struct to represent an image
typedef struct {
    int width;
    int height;
    RGBPixel *rgb;
} Image;

// Image creation and manipulation functions
Image *create_image(int width, int height);
void free_image(Image *img);

// Image processing function prototypes
void horizontal_flip(Image * img);
void vertical_flip(Image *img);
void scale(Image *img, int new_width, int new_height, int is_bilinear);
void rotate_image(Image *img, double degrees, int n);
void edge_detection(Image *img);
void vertical_shear(Image *img, float offset);
void horizontal_shear(Image *img, float offset);
void crop(Image *img, int start_x, int start_y, int end_x, int end_y);
void linear_mapping(Image *img, float a, float b);
void power_mapping(Image *img, float gamma);
void min_filter(Image *img);
void median_filter(Image *img);
void max_filter(Image *img);
void convolution(Image *img, int *kernel, int n, int m);
#endif
