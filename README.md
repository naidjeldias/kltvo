# **KLTVO**

KLTVO is a real-time Stereo Visual Odometry (S-VO) system that relies on a circular matching approach for
feature selection using spatial and temporal information. The process combines the Illumination Normalized SAD metric for stereo matching and the KLT algorithm for feature tracking. In both approaches, we explore the epipolar geometry constraints to get a fast and accurate feature correspondence.

## **Related Publications**
DIAS, N. J. B. KLTVO: Algoritmo de Odometria Visual estéreo baseada em seleção de pontos chaves pela imposição das restrições da geometria epipolar. 2020. 103 f. Dissertação (Mestrado em Ciência da Computação) - Universidade Federal de Goiás, Goiânia, 2020. [PDF](https://repositorio.bc.ufg.br/tede/bitstream/tede/10746/3/Disserta%c3%a7%c3%a3o%20-%20Nigel%20Joseph%20Bandeira%20Dias%20-%202020.pdf) (Portuguese)

N. Dias and G. Laureano, "Accurate Stereo Visual Odometry Based on Keypoint Selection," 2019 Latin American Robotics Symposium (LARS), 2019 Brazilian Symposium on Robotics (SBR) and 2019 Workshop on Robotics in Education (WRE), 2019, pp. 74-79, doi: 10.1109/LARS-SBR-WRE48964.2019.00021. [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9018624)

If you use KLTVO in an academic work, please cite:
```
@INPROCEEDINGS{9018624,
  author={Dias, Nigel and Laureano, Gustavo},
  booktitle={2019 Latin American Robotics Symposium (LARS), 2019 Brazilian Symposium on Robotics (SBR) and 2019 Workshop on Robotics in Education (WRE)}, 
  title={Accurate Stereo Visual Odometry Based on Keypoint Selection}, 
  year={2019},
  volume={},
  number={},
  pages={74-79},
  doi={10.1109/LARS-SBR-WRE48964.2019.00021}}
```

# **Install**
## Dependecies
The code has been tested with the dependencies below. Still, it could be run with different versions, except the GCC compiler version which throws segmentation faults with higher versions due to the std::thread library. 
- [OpenCV](https://github.com/opencv/opencv) 4.0.0
- [Eigen](https://github.com/libigl/eigen) 3
- [Pangolin](https://github.com/stevenlovegrove/Pangolin) 0.4
- Python 3.8 (numpy and matplotlib)
- g++/gcc 5 and c++11
- Ubuntu 20.04

## Build
Please make sure you have installed all required dependecies list above.
```
$ git clone https://github.com/naidjeldias/kltvo.git KLTVO
$ cd KLTVO
$ mkdir -p build
$ cd build
$ cmake ..
$ make
```

## Runing the examples
We provide examples to run the KLTVO system in the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and in the [EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets), before running the examples make sure that it has been downloaded.

### KITTI stereo example
```
Usage: ./stereo_kitti <SEQUENCE_PATH> <SEQUENCE_ID>
Example: ./stereo_kitti ~/dataset/sequences/00/ 00
```

### EuRoC stereo example
```
Usage: ./stereo_euroc <SEQUENCE_PATH> <SEQUENCE_ID>
Example: ./stereo_euroc ~/MH_02_easy/mav0/ MH02
```

# **Docker**
We highly recommend you use docker instead of setting up the environment directly on your machine. For more information about docker and how to get started read the [documentation](https://docs.docker.com/get-started/). Additionally, to enable the x11 forwarding from the container for the application display we use rocker which allows running docker images with customized local support injected for things like Nvidia support. For more information and the installation process follow [their guidelines on github](https://github.com/osrf/rocker).

In summary to run the application with docker you need to install the following dependencies: 
- [docker](https://docs.docker.com/get-started/)
- [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) - may not be mandatory
- [rocker](https://github.com/osrf/rocker)
  
Once you have installed docker and all the dependecies use the following commands to run the container:
## Nvidia graphic cards support
```
rocker  --x11 \
        --nvidia  \
        --name kltvo \
        --oyr-run-arg " -v <HOST_DATASET_PATH>:/data/"  \
        naidjeldias/kltvo bash
```

## Intel integrated graphics support
```
rocker  --x11 \
        --devices /dev/dri \
        --name kltvo \
        --oyr-run-arg " -v <HOST_DATASET_PATH>:/data/" \
        naidjeldias/kltvo bash
```

**Note**: Do not forget to replace `<HOST_DATASET_PATH>` with the dataset path on your machine. 

The commands above will start a bash session inside the container so you can run the examples as mentioned earlier but now the data will be hosted on `/data/` path:

```
./stereo_kitti /data/<KITTI_DATASET_FOLDER>/sequences/00/ 00
./stereo_euroc /data/<EuRoC_DATASET_FOLDER>/mav0/ MH02
```

<img src="docs/kltvo.gif" > 