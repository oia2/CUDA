#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>

__global__ void computeChannels(const uchar* img, uchar* blueChannel, uchar* yellowChannel, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int offset = y * cols + x;

        uchar redv = img[offset * 3 + 2];
        uchar greenv = img[offset * 3 + 1];
        uchar bluev = img[offset * 3];

        // Вычисление модифицированного синего канала
        uchar Bv = bluev - (greenv + bluev) / 2;

        // Вычисление модифицированного желтого канала
        uchar Yv = redv + greenv - 2 * (abs(redv - greenv) + bluev);

        // Запись результатов в соответствующие Mat-объекты
        blueChannel[offset] = Bv;
        yellowChannel[offset] = Yv;
    }
}

int main() {
    for (int i = 1; i <= 10; i++) {
        cv::Mat image = cv::imread("../src/giena_1024x768.jpg");

        if (image.empty()) {
            std::cout << "Failed to upload image" << std::endl;
            return -1;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Создание двух пустых Mat-объектов для модифицированных синего и желтого каналов
        cv::Mat modifiedBlueChannel = cv::Mat::zeros(image.size(), CV_8U);
        cv::Mat modifiedYellowChannel = cv::Mat::zeros(image.size(), CV_8U);


        uchar* d_img;
        cudaMalloc((void**)&d_img, image.rows * image.cols * 3);
        cudaMemcpy(d_img, image.data, image.rows * image.cols * 3, cudaMemcpyHostToDevice);

        uchar* d_blueChannel, * d_yellowChannel;
        cudaMalloc((void**)&d_blueChannel, image.rows * image.cols);
        cudaMalloc((void**)&d_yellowChannel, image.rows * image.cols);

        const dim3 block(32, 32);
        const dim3 grid((image.cols + block.x - 1) / block.x, (image.rows + block.y - 1) / block.y);


        // Запуск ядра на устройстве
        computeChannels << <grid, block >> > (d_img, d_blueChannel, d_yellowChannel, image.rows, image.cols);
        cudaDeviceSynchronize();

        // Копирование результатов обратно
        cudaMemcpy(modifiedBlueChannel.data, d_blueChannel, image.rows * image.cols, cudaMemcpyDeviceToHost);
        cudaMemcpy(modifiedYellowChannel.data, d_yellowChannel, image.rows * image.cols, cudaMemcpyDeviceToHost);
        // Замер времени заканчивается
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << i << ". Time " << duration.count() << " ms" << std::endl;
        // Освобождение выделенной памяти на устройстве
        cudaFree(d_img);
        cudaFree(d_blueChannel);
        cudaFree(d_yellowChannel);

        // Сохранение изображений в файлы
        cv::imwrite("../modified_blue_channel.jpg", modifiedBlueChannel);
        cv::imwrite("../modified_yellow_channel.jpg", modifiedYellowChannel);
    }
    return 0;
}
