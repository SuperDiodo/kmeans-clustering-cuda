# kmeans-clustering-cuda
CUDA implementation of Kmeans algorithm for image clustering.



### Compile

```shell
mkdir build
cd build
cmake ..
make
```



### Run

```shell
cd build

# non optimized version
./image_segmentation ../data/beach.jpg output.png 3

# optimized version
./image_segmentation_opt ../data/beach.jpg output.png 3

# resize the input/ouput by an half
./image_segmentation ../data/beach.jpg output.png 3 0.5
```
![alt text](https://github.com/SuperDiodo/kmeans-clustering-cuda/blob/main/output/lake_k3.png)
