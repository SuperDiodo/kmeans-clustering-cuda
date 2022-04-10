#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <math.h>

#include <opencv2/opencv.hpp>

__constant__ int K;

#define TH 0.001
#define MAX_ITERATION 500

/* DEVICE */

__global__ void association_kernel(const float3 *img, const size_t width, const size_t height, const float3 *centroids, int *result)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t i = x + y * width;

    if (x >= width || y >= height)
        return;

    float min_distance = 100;
    int best_cluster_index = 2;

    for (int c = 0; c < K; c++)
    {
        const float dred = img[i].z - centroids[c].z;
        const float dblue = img[i].x - centroids[c].x;
        const float dgreen = img[i].y - centroids[c].y;
        const float distance = sqrtf(dred * dred + dblue * dblue + dgreen * dgreen);

        if (distance < min_distance)
        {
            best_cluster_index = c;
            min_distance = distance;
        }
    }

    result[i] = best_cluster_index;
}

__global__ void sum_kernel(const float3 *img, const size_t width, const size_t height, const int *associations, float4 *result)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t i = x + y * width;

    if (x >= width || y >= height)
        return;

    atomicAdd(&(result[associations[i]].x), img[i].x);
    atomicAdd(&(result[associations[i]].y), img[i].y);
    atomicAdd(&(result[associations[i]].z), img[i].z);
    atomicAdd(&(result[associations[i]].w), 1);
}

__global__ void update_kernel(const float4 *centroid_infos, const size_t width, float3 *result)
{

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t i = x + y * K;

    if (i >= K)
        return;

    if (centroid_infos[i].w > 0)
    {
        result[i].x = centroid_infos[i].x / centroid_infos[i].w;
        result[i].y = centroid_infos[i].y / centroid_infos[i].w;
        result[i].z = centroid_infos[i].z / centroid_infos[i].w;
    }
}

/* HOST */

void initialization(const cv::Mat &image, const size_t width, const size_t height, const int &k, std::vector<float3> &centroids)
{
    for (auto &c : centroids)
    {
        int r_row = rand() % height;
        int r_col = rand() % width;
        c.x = image.at<cv::Vec3f>(r_row, r_col)[0];
        c.y = image.at<cv::Vec3f>(r_row, r_col)[1];
        c.z = image.at<cv::Vec3f>(r_row, r_col)[2];
    }
}

bool noUpdateNeeded(const std::vector<float3> &old_centroids, const std::vector<float3> &new_centroids)
{

    float max_distance = 0.0;
    for (int i = 0; i < old_centroids.size(); i++)
    {
        const float dred = old_centroids[i].z - new_centroids[i].z;
        const float dblue = old_centroids[i].x - new_centroids[i].x;
        const float dgreen = old_centroids[i].y - new_centroids[i].y;
        const float distance = sqrtf(dred * dred + dblue * dblue + dgreen * dgreen);

        if (distance > max_distance)
            max_distance = distance;
    }

    std::cout << "distance from previous centroids: " << max_distance << std::endl;

    if (max_distance < TH)
        return true;
    return false;
}

void printImage(const cv::Mat &image, const std::vector<int> &association_kernel, const std::string &filename, const std::vector<float3> color_palette)
{

    cv::Mat converted;
    image.convertTo(converted, CV_8UC3, 255.0f);

    for (int j = 0; j < converted.rows; j++)
    {
        for (int i = 0; i < converted.cols; i++)
        {
            const int elem = j * converted.size().width + i;
            converted.at<cv::Vec3b>(j, i)[0] = color_palette[association_kernel[elem]].x*255;
            converted.at<cv::Vec3b>(j, i)[1] = color_palette[association_kernel[elem]].y*255;
            converted.at<cv::Vec3b>(j, i)[2] = color_palette[association_kernel[elem]].z*255;
        }
    }

    cv::imwrite(filename, converted);
    cv::imshow("result", converted);
    cv::waitKey(0);
}

void CheckCudaError(const std::string reason)
{
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Cuda error: %s (%s)\n", cudaGetErrorString(err), reason.c_str());
        exit(1);
    }
}

int main(int argc, char **argv)
{

    if (argc < 4)
    {
        std::cerr << "Usage 1: ./image_segmentation input_image output_image num_cluster" << std::endl;
        std::cerr << "Usage 2: ./image_segmentation input_image output_image num_cluster %_resize" << std::endl;
        return 1;
    }

    srand(time(NULL));

    const std::string input_filename = argv[1];
    const std::string output_filename = argv[2];

    cv::Mat input_image = cv::imread(input_filename, cv::IMREAD_COLOR);

    if (!input_image.data)
    {
        std::cerr << "Cannot load image " << input_filename << std::endl;
        return 1;
    }

    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);

    if(argc == 5){
        cv::resize(input_image, input_image, cv::Size(), atof(argv[4]), atof(argv[4]));
    }

    cv::Mat blurredImage;
    cv::GaussianBlur( input_image, blurredImage, cv::Size( 9, 9 ), 1.0);

    const size_t width = blurredImage.cols;
    const size_t height = blurredImage.rows;

    float3 *device_img;
    cudaMalloc(&device_img, width * height * sizeof(float3));
    CheckCudaError("malloc image");

    cudaMemcpy(device_img, blurredImage.data, width * height * sizeof(float3), cudaMemcpyHostToDevice);
    CheckCudaError("upload image");

    int k = atoi(argv[3]);
    cudaMemcpyToSymbol(K, &k, sizeof(int));
    CheckCudaError("upload K");

    float3 *device_centroids;
    cudaMalloc(&device_centroids, k * sizeof(float3));
    CheckCudaError("malloc centroids");

    std::vector<float3> host_centroids(k);

    initialization(blurredImage, width, height, k, host_centroids);

    cudaMemcpy(device_centroids, host_centroids.data(), k * sizeof(float3), cudaMemcpyHostToDevice);
    CheckCudaError("upload initialized centroids");

    int *device_associations;
    cudaMalloc(&device_associations, width * height * sizeof(int));
    CheckCudaError("malloc associations");

    float4 *device_centroid_sums;
    cudaMalloc(&device_centroid_sums, width * height * sizeof(float4));
    CheckCudaError("malloc centroid sums");

    float *device_equal_check_distance;
    cudaMalloc(&device_equal_check_distance, sizeof(float));
    CheckCudaError("malloc equal check");

    std::vector<int> start_association(width * height);
    for (auto &a : start_association)
        a = -1;

    bool done = false;
    int iteration = 0;

    while (!done && iteration < MAX_ITERATION)
    {

        const int block_height = 32;
        const int block_width = 32;
        dim3 blockDim(block_width, block_height);
        dim3 gridDim((width + (block_width - 1)) / block_width, (height + (block_height - 1)) / block_height);
        association_kernel<<<gridDim, blockDim>>>(device_img, width, height, device_centroids, device_associations);
        CheckCudaError("associate pixel-centroids");

        cudaMemset(device_centroid_sums,0,width * height * sizeof(float4));
        sum_kernel<<<gridDim, blockDim>>>(device_img, width, height, device_associations, device_centroid_sums);
        CheckCudaError("compute centroids sums");

        update_kernel<<<1, k>>>(device_centroid_sums, width, device_centroids);
        CheckCudaError("update centroids");

        std::vector<float3> host_updated_centroids(k);
        cudaMemcpy(host_updated_centroids.data(), device_centroids, k * sizeof(float3), cudaMemcpyDeviceToHost);
        CheckCudaError("download updated centroids");

        if (noUpdateNeeded(host_centroids, host_updated_centroids))
            done = true;

        host_centroids = host_updated_centroids;

        iteration++;
    }

    std::cout << "done in " << iteration << " iterations" << std::endl;

    std::vector<int> host_associations(width * height);
    cudaMemcpy(host_associations.data(), device_associations, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    CheckCudaError("download associations");

    printImage(blurredImage, host_associations, output_filename, host_centroids);

    cudaFree(device_img);
    cudaFree(device_centroids);
    cudaFree(device_associations);
    cudaFree(device_centroid_sums);
    cudaFree(device_equal_check_distance);

    return 0;
}
