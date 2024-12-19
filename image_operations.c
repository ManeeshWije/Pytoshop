#include "image_operations.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Image *create_image(int width, int height) {
    Image *img = malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->rgb = malloc(width * height * sizeof(RGBPixel));
    return img;
}

// Comparison function for qsort
int compare_int(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

void free_image(Image *img) {
    if (img) {
        free(img->rgb);
        free(img);
    }
}

void horizontal_flip(Image *img) {
    if (!img)
        return;

    for (int j = 0; j < img->height; j++) {
        for (int i = 0; i < img->width / 2; i++) {
            RGBPixel temp = img->rgb[j * img->width + i];
            img->rgb[j * img->width + i] = img->rgb[j * img->width + (img->width - i - 1)];
            img->rgb[j * img->width + (img->width - i - 1)] = temp;
        }
    }
}

void vertical_flip(Image *img) {
    if (!img)
        return;

    for (int i = 0; i < img->width; i++) {
        for (int j = 0; j < img->height / 2; j++) {
            RGBPixel temp = img->rgb[j * img->width + i];
            img->rgb[j * img->width + i] = img->rgb[(img->height - j - 1) * img->width + i];
            img->rgb[(img->height - j - 1) * img->width + i] = temp;
        }
    }
}

void scale(Image *img, int new_width, int new_height, int is_bilinear) {
    if (!img || new_width <= 0 || new_height <= 0)
        return;

    // Create a new image with the desired dimensions
    Image *resized = create_image(new_width, new_height);
    if (!resized) {
        return;
    }

    if (!is_bilinear) {
        // Nearest Neighbor Interpolation
        for (int y = 0; y < new_height; y++) {
            for (int x = 0; x < new_width; x++) {
                // Map new image coordinates to original image
                int src_x = (int)(x * img->width / new_width);
                int src_y = (int)(y * img->height / new_height);

                // Ensure we don't go out of bounds
                src_x = (src_x < 0) ? 0 : (src_x >= img->width) ? img->width - 1 : src_x;
                src_y = (src_y < 0) ? 0 : (src_y >= img->height) ? img->height - 1 : src_y;

                // Copy pixel from source to destination
                resized->rgb[y * new_width + x] = img->rgb[src_y * img->width + src_x];
            }
        }
    } else {
        // Bilinear Interpolation
        for (int y = 0; y < new_height; y++) {
            for (int x = 0; x < new_width; x++) {
                // Map new image coordinates to original image
                float src_x = (float)x * img->width / new_width;
                float src_y = (float)y * img->height / new_height;

                // Find surrounding pixel coordinates
                int x0 = (int)src_x;
                int y0 = (int)src_y;
                int x1 = (x0 + 1 < img->width) ? x0 + 1 : img->width - 1;
                int y1 = (y0 + 1 < img->height) ? y0 + 1 : img->height - 1;

                // Calculate interpolation weights
                float wx = src_x - x0;
                float wy = src_y - y0;

                // Get surrounding pixels
                RGBPixel p00 = img->rgb[y0 * img->width + x0];
                RGBPixel p01 = img->rgb[y0 * img->width + x1];
                RGBPixel p10 = img->rgb[y1 * img->width + x0];
                RGBPixel p11 = img->rgb[y1 * img->width + x1];

                // Interpolate each color channel
                RGBPixel interpolated;
                interpolated.r = (unsigned char)(p00.r * (1 - wx) * (1 - wy) + p01.r * wx * (1 - wy) +
                                                 p10.r * (1 - wx) * wy + p11.r * wx * wy);
                interpolated.g = (unsigned char)(p00.g * (1 - wx) * (1 - wy) + p01.g * wx * (1 - wy) +
                                                 p10.g * (1 - wx) * wy + p11.g * wx * wy);
                interpolated.b = (unsigned char)(p00.b * (1 - wx) * (1 - wy) + p01.b * wx * (1 - wy) +
                                                 p10.b * (1 - wx) * wy + p11.b * wx * wy);

                resized->rgb[y * new_width + x] = interpolated;
            }
        }
    }

    free(img->rgb);
    img->rgb = malloc(new_width * new_height * sizeof(RGBPixel));
    if (!img->rgb) {
        free_image(resized);
        return;
    }

    // Copy resized data back to img
    memcpy(img->rgb, resized->rgb, new_width * new_height * sizeof(RGBPixel));
    img->width = new_width;
    img->height = new_height;

    // Free the temporary resized image
    free_image(resized);
}

void edge_detection(Image *img) {
    // Validate input
    if (!img || !img->rgb || img->width < 3 || img->height < 3) {
        return; // Not enough data to perform edge detection
    }

    // Create a new image to store the result
    Image *result = create_image(img->width, img->height);
    if (!result) {
        return;
    }

    // Sobel kernels for x and y directions
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Iterate through each pixel (excluding border pixels)
    for (int y = 1; y < img->height - 1; y++) {
        for (int x = 1; x < img->width - 1; x++) {
            int gx_r = 0, gx_g = 0, gx_b = 0;
            int gy_r = 0, gy_g = 0, gy_b = 0;

            // Apply Sobel kernels
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    // Boundary-safe access
                    int safe_y = (y + dy < 0) ? 0 : (y + dy >= img->height) ? img->height - 1 : y + dy;
                    int safe_x = (x + dx < 0) ? 0 : (x + dx >= img->width) ? img->width - 1 : x + dx;

                    RGBPixel pixel = img->rgb[safe_y * img->width + safe_x];

                    int kernel_x = sobel_x[dy + 1][dx + 1];
                    int kernel_y = sobel_y[dy + 1][dx + 1];

                    // Compute gradient for each channel
                    gx_r += pixel.r * kernel_x;
                    gx_g += pixel.g * kernel_x;
                    gx_b += pixel.b * kernel_x;

                    gy_r += pixel.r * kernel_y;
                    gy_g += pixel.g * kernel_y;
                    gy_b += pixel.b * kernel_y;
                }
            }

            // Compute gradient magnitude
            int magnitude_r = (int)sqrt(gx_r * gx_r + gy_r * gy_r);
            int magnitude_g = (int)sqrt(gx_g * gx_g + gy_g * gy_g);
            int magnitude_b = (int)sqrt(gx_b * gx_b + gy_b * gy_b);

            // Clamp values to 0-255
            magnitude_r = (magnitude_r > 255) ? 255 : (magnitude_r < 0) ? 0 : magnitude_r;
            magnitude_g = (magnitude_g > 255) ? 255 : (magnitude_g < 0) ? 0 : magnitude_g;
            magnitude_b = (magnitude_b > 255) ? 255 : (magnitude_b < 0) ? 0 : magnitude_b;

            // Store in result image
            result->rgb[y * img->width + x].r = magnitude_r;
            result->rgb[y * img->width + x].g = magnitude_g;
            result->rgb[y * img->width + x].b = magnitude_b;
        }
    }

    // Clear border pixels to 0
    for (int x = 0; x < img->width; x++) {
        // Top and bottom rows
        result->rgb[x].r = result->rgb[x].g = result->rgb[x].b = 0;
        result->rgb[(img->height - 1) * img->width + x].r = 0;
        result->rgb[(img->height - 1) * img->width + x].g = 0;
        result->rgb[(img->height - 1) * img->width + x].b = 0;
    }

    for (int y = 0; y < img->height; y++) {
        // Left and right columns
        result->rgb[y * img->width].r = result->rgb[y * img->width].g = result->rgb[y * img->width].b = 0;
        result->rgb[y * img->width + (img->width - 1)].r = 0;
        result->rgb[y * img->width + (img->width - 1)].g = 0;
        result->rgb[y * img->width + (img->width - 1)].b = 0;
    }

    // Replace original image with edge-detected image
    memcpy(img->rgb, result->rgb, img->width * img->height * sizeof(RGBPixel));
    free_image(result);
}

void rotate_image(Image *img, double degrees, int n) {
    if (!img || img->width <= 0 || img->height <= 0)
        return;

    double radians = degrees * M_PI / 180.0;
    double cos_theta = cos(radians);
    double sin_theta = sin(radians);

    // Calculate new dimensions
    int new_width = (int)(fabs(img->width * cos_theta) + fabs(img->height * sin_theta));
    int new_height = (int)(fabs(img->height * cos_theta) + fabs(img->width * sin_theta));

    // Create new image and oversampled image
    Image *new_img = create_image(new_width, new_height);
    if (!new_img)
        return;

    int oversampled_width = n * img->width;
    int oversampled_height = n * img->height;
    Image *oversampled_img = create_image(oversampled_width, oversampled_height);
    if (!oversampled_img) {
        free_image(new_img);
        return;
    }

    // Fill oversampled image with repeated pixels
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            RGBPixel pixel = img->rgb[y * img->width + x];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    oversampled_img->rgb[(n * y + i) * oversampled_width + (n * x + j)] = pixel;
                }
            }
        }
    }

    // Rotate oversampled image
    int cx = img->width / 2;
    int cy = img->height / 2;
    int new_cx = new_width / 2;
    int new_cy = new_height / 2;

    for (int y = 0; y < oversampled_height; y++) {
        for (int x = 0; x < oversampled_width; x++) {
            double src_x = (x / (double)n) - cx;
            double src_y = (y / (double)n) - cy;
            int new_x = (int)(round(src_x * cos_theta - src_y * sin_theta) + new_cx);
            int new_y = (int)(round(src_x * sin_theta + src_y * cos_theta) + new_cy);

            if (new_x >= 0 && new_x < new_width && new_y >= 0 && new_y < new_height) {
                new_img->rgb[new_y * new_width + new_x] = oversampled_img->rgb[y * oversampled_width + x];
            }
        }
    }

    // Update the original image
    RGBPixel *new_rgb = realloc(img->rgb, new_width * new_height * sizeof(RGBPixel));
    if (!new_rgb) {
        free_image(new_img);
        free_image(oversampled_img);
        return;
    }

    img->rgb = new_rgb;
    memcpy(img->rgb, new_img->rgb, new_width * new_height * sizeof(RGBPixel));
    img->width = new_width;
    img->height = new_height;

    free_image(oversampled_img);
    free_image(new_img);
}

void horizontal_shear(Image *img, float offset) {
    if (!img || img->width <= 0 || img->height <= 0)
        return;

    int max_shift = abs((int)(img->height * offset));
    int new_width = img->width + max_shift;
    int new_height = img->height;

    // Create a new image with the new dimensions
    Image *sheared_img = create_image(new_width, new_height);
    if (!sheared_img)
        return;

    // Initialize the new image to a default color (e.g., black)
    for (int j = 0; j < new_height; j++) {
        for (int i = 0; i < new_width; i++) {
            sheared_img->rgb[j * new_width + i] = (RGBPixel){0, 0, 0};
        }
    }

    // Perform the shear operation
    for (int j = 0; j < img->height; j++) {
        for (int i = 0; i < img->width; i++) {
            int new_i = i + (int)(j * offset);
            if (new_i >= 0 && new_i < new_width) {
                sheared_img->rgb[j * new_width + new_i] = img->rgb[j * img->width + i];
            }
        }
    }

    // Replace the original image with the sheared image
    free(img->rgb);
    img->rgb = malloc(new_width * new_height * sizeof(RGBPixel));
    if (!img->rgb) {
        free_image(sheared_img);
        return;
    }

    memcpy(img->rgb, sheared_img->rgb, new_width * new_height * sizeof(RGBPixel));
    img->width = new_width;
    img->height = new_height;

    free_image(sheared_img);
}

void vertical_shear(Image *img, float offset) {
    if (!img || img->width <= 0 || img->height <= 0)
        return;

    int max_shift = abs((int)(img->width * offset));
    int new_width = img->width;
    int new_height = img->height + max_shift;

    // Create a new image with the new dimensions
    Image *sheared_img = create_image(new_width, new_height);
    if (!sheared_img)
        return;

    // Initialize the new image to a default color (e.g., black)
    for (int j = 0; j < new_height; j++) {
        for (int i = 0; i < new_width; i++) {
            sheared_img->rgb[j * new_width + i] = (RGBPixel){0, 0, 0};
        }
    }

    // Perform the shear operation
    for (int j = 0; j < img->height; j++) {
        for (int i = 0; i < img->width; i++) {
            int new_j = j + (int)(i * offset);
            if (new_j >= 0 && new_j < new_height) {
                sheared_img->rgb[new_j * new_width + i] = img->rgb[j * img->width + i];
            }
        }
    }

    // Replace the original image with the sheared image
    free(img->rgb);
    img->rgb = malloc(new_width * new_height * sizeof(RGBPixel));
    if (!img->rgb) {
        free_image(sheared_img);
        return;
    }

    memcpy(img->rgb, sheared_img->rgb, new_width * new_height * sizeof(RGBPixel));
    img->width = new_width;
    img->height = new_height;

    free_image(sheared_img);
}

void crop(Image *img, int start_x, int start_y, int end_x, int end_y, int mode) {
    if (!img || img->width <= 0 || img->height <= 0)
        return;

    printf("MODE %d\n", mode);

    // Iterate through all pixels in the image
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            // If the current pixel is within the crop region
            if (x >= start_x && x < end_x && y >= start_y && y < end_y) {
                // Inside the crop region, keep the pixel as is
                img->rgb[y * img->width + x] = img->rgb[y * img->width + x];
            } else {
                // Outside the crop region, apply the selected mode
                if (mode == 0) {
                    // Black background (out-of-bounds pixels are black)
                    img->rgb[y * img->width + x] = (RGBPixel){0, 0, 0};
                } else if (mode == 1) {
                    // Circular indexing (wrap around the image)
                    int orig_x = (x + img->width) % img->width;   // Wrap around horizontally
                    int orig_y = (y + img->height) % img->height; // Wrap around vertically
                    img->rgb[y * img->width + x] = img->rgb[orig_y * img->width + orig_x];
                } else if (mode == 2) {
                    // Reflected indexing (mirror the image)
                    int orig_x = x;
                    int orig_y = y;

                    // Reflect x-coordinate
                    if (x < 0) {
                        orig_x = -x;
                    } else if (x >= img->width) {
                        orig_x = img->width - (x - img->width + 1);
                    }

                    // Reflect y-coordinate
                    if (y < 0) {
                        orig_y = -y;
                    } else if (y >= img->height) {
                        orig_y = img->height - (y - img->height + 1);
                    }

                    img->rgb[y * img->width + x] = img->rgb[orig_y * img->width + orig_x];
                }
            }
        }
    }
}

void linear_mapping(Image *img, float a, float b) {
    if (!img || img->width <= 0 || img->height <= 0) {
        return;
    }

    Image *newImage = create_image(img->width, img->height);
    if (!newImage) {
        return;
    }

    for (int i = 0; i < img->height; i++) {
        for (int j = 0; j < img->width; j++) {
            RGBPixel *pixel = &img->rgb[i * img->width + j];
            RGBPixel *newPixel = &newImage->rgb[i * img->width + j];

            if (pixel->r == pixel->g && pixel->r == pixel->b) {
                // If grayscale (same value for r, g, b), apply the grayscale transformation
                unsigned char greyValue = pixel->r; // All channels are the same for grayscale
                unsigned char newGreyValue = (unsigned char)(a * greyValue + b);
                newPixel->r = newPixel->g = newPixel->b = newGreyValue;
            } else {
                // RGB transformation
                newPixel->r = (unsigned char)fminf(fmaxf(a * pixel->r + b, 0), 255);
                newPixel->g = (unsigned char)fminf(fmaxf(a * pixel->g + b, 0), 255);
                newPixel->b = (unsigned char)fminf(fmaxf(a * pixel->b + b, 0), 255);
            }
        }
    }

    // Replace the original image with the new image
    memcpy(img->rgb, newImage->rgb, newImage->width * newImage->height * sizeof(RGBPixel));
    free_image(newImage);
}

void power_mapping(Image *img, float gamma) {
    if (!img || img->width <= 0 || img->height <= 0) {
        return;
    }

    Image *newImage = create_image(img->width, img->height);
    if (!newImage) {
        return;
    }

    for (int i = 0; i < img->height; i++) {
        for (int j = 0; j < img->width; j++) {
            RGBPixel *pixel = &img->rgb[i * img->width + j];
            RGBPixel *newPixel = &newImage->rgb[i * img->width + j];

            if (pixel->r == pixel->g && pixel->r == pixel->b) {
                // If grayscale (same value for r, g, b), apply the grayscale transformation
                unsigned char greyValue = pixel->r; // All channels are the same for grayscale
                unsigned char newGreyValue = (unsigned char)(255 * pow((float)greyValue / 255.0, gamma));
                newPixel->r = newPixel->g = newPixel->b = newGreyValue;
            } else {
                // Apply power-law transformation (gamma correction) for RGB channels
                newPixel->r = (unsigned char)fminf(fmaxf(255 * pow((float)pixel->r / 255.0, gamma), 0), 255);
                newPixel->g = (unsigned char)fminf(fmaxf(255 * pow((float)pixel->g / 255.0, gamma), 0), 255);
                newPixel->b = (unsigned char)fminf(fmaxf(255 * pow((float)pixel->b / 255.0, gamma), 0), 255);
            }
        }
    }

    // Replace the original image with the new image
    memcpy(img->rgb, newImage->rgb, newImage->width * newImage->height * sizeof(RGBPixel));
    free_image(newImage);
}

void min_filter(Image *img) {
    if (!img || img->width < 3 || img->height < 3)
        return;

    // Create a new image to store the result
    Image *result = create_image(img->width, img->height);
    if (!result)
        return;

    for (int y = 1; y < img->height - 1; y++) {
        for (int x = 1; x < img->width - 1; x++) {
            int min_r = 255, min_g = 255, min_b = 255;

            // Iterate through the 3x3 neighborhood
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    // Get the neighboring pixel (x+dx, y+dy)
                    int neighbor_x = x + dx;
                    int neighbor_y = y + dy;
                    RGBPixel pixel = img->rgb[neighbor_y * img->width + neighbor_x];

                    // For RGB images, update each channel
                    min_r = (pixel.r < min_r) ? pixel.r : min_r;
                    min_g = (pixel.g < min_g) ? pixel.g : min_g;
                    min_b = (pixel.b < min_b) ? pixel.b : min_b;
                }
            }

            // Assign the minimum values to the corresponding pixel in the result image
            result->rgb[y * img->width + x].r = min_r;
            result->rgb[y * img->width + x].g = min_g;
            result->rgb[y * img->width + x].b = min_b;
        }
    }

    // Copy the result back to the original image
    memcpy(img->rgb, result->rgb, img->width * img->height * sizeof(RGBPixel));
    free_image(result);
}

void median_filter(Image *img) {
    if (!img || img->width < 3 || img->height < 3)
        return;

    // Create a new image to store the result
    Image *result = create_image(img->width, img->height);
    if (!result)
        return;

    for (int y = 1; y < img->height - 1; y++) {
        for (int x = 1; x < img->width - 1; x++) {
            int r_values[9], g_values[9], b_values[9];
            int index = 0;

            // Collect all the RGB values from the 3x3 neighborhood
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int neighbor_x = x + dx;
                    int neighbor_y = y + dy;
                    RGBPixel pixel = img->rgb[neighbor_y * img->width + neighbor_x];

                    r_values[index] = pixel.r;
                    g_values[index] = pixel.g;
                    b_values[index] = pixel.b;
                    index++;
                }
            }

            // Sort each channel's values to find the median
            qsort(r_values, 9, sizeof(int), compare_int);
            qsort(g_values, 9, sizeof(int), compare_int);
            qsort(b_values, 9, sizeof(int), compare_int);

            // The median is the middle value of the sorted array
            result->rgb[y * img->width + x].r = r_values[4];
            result->rgb[y * img->width + x].g = g_values[4];
            result->rgb[y * img->width + x].b = b_values[4];
        }
    }

    // Copy the result back to the original image
    memcpy(img->rgb, result->rgb, img->width * img->height * sizeof(RGBPixel));
    free_image(result);
}

void max_filter(Image *img) {
    if (!img || img->width < 3 || img->height < 3)
        return;

    // Create a new image to store the result
    Image *result = create_image(img->width, img->height);
    if (!result)
        return;

    for (int y = 1; y < img->height - 1; y++) {
        for (int x = 1; x < img->width - 1; x++) {
            int max_r = 0, max_g = 0, max_b = 0;

            // Iterate through the 3x3 neighborhood
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int neighbor_x = x + dx;
                    int neighbor_y = y + dy;
                    RGBPixel pixel = img->rgb[neighbor_y * img->width + neighbor_x];

                    // For RGB images, update each channel
                    max_r = (pixel.r > max_r) ? pixel.r : max_r;
                    max_g = (pixel.g > max_g) ? pixel.g : max_g;
                    max_b = (pixel.b > max_b) ? pixel.b : max_b;
                }
            }

            // Assign the maximum values to the corresponding pixel in the result image
            result->rgb[y * img->width + x].r = max_r;
            result->rgb[y * img->width + x].g = max_g;
            result->rgb[y * img->width + x].b = max_b;
        }
    }

    // Copy the result back to the original image
    memcpy(img->rgb, result->rgb, img->width * img->height * sizeof(RGBPixel));
    free_image(result);
}

void convolution(Image *img, int *kernel, int n, int m) {
    if (!img || !kernel || n <= 0 || m <= 0) {
        return;
    }

    // Create a temporary image to store the result
    Image *result = create_image(img->width, img->height);
    if (!result) {
        return;
    }

    // Zero-padding the image (for simplicity)
    int padding_x = n / 2;
    int padding_y = m / 2;

    // Loop through each pixel in the image
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {

            // Initialize color values for the new pixel
            int sum_r = 0, sum_g = 0, sum_b = 0;

            // Apply the kernel
            for (int ky = -padding_y; ky <= padding_y; ky++) {
                for (int kx = -padding_x; kx <= padding_x; kx++) {
                    // Get the corresponding pixel in the image (with bounds checking)
                    int px = x + kx;
                    int py = y + ky;

                    // Ensure the pixel is within bounds
                    if (px >= 0 && px < img->width && py >= 0 && py < img->height) {
                        RGBPixel pixel = img->rgb[py * img->width + px];

                        // Get the kernel value at the current position
                        int kernel_value = kernel[(ky + padding_y) * m + (kx + padding_x)];

                        // Multiply the kernel value with the pixel color and accumulate
                        sum_r += pixel.r * kernel_value;
                        sum_g += pixel.g * kernel_value;
                        sum_b += pixel.b * kernel_value;
                    }
                }
            }

            // Normalize the results (clamp to the valid range for colors)
            result->rgb[y * img->width + x].r = (unsigned char)fmin(fmax(sum_r, 0), 255);
            result->rgb[y * img->width + x].g = (unsigned char)fmin(fmax(sum_g, 0), 255);
            result->rgb[y * img->width + x].b = (unsigned char)fmin(fmax(sum_b, 0), 255);
        }
    }

    // Copy the result back to the original image
    memcpy(img->rgb, result->rgb, img->width * img->height * sizeof(RGBPixel));
    free_image(result);
}
