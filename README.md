# FWI
This project is time domain full waveform inversion of elastic wave equation by using CUDA.

## Introduction

The purpose of Full Waveform Inversion (FWI) is to obtain the subsurface model parameters by minimizing the objective functional between the observed and synthetic seismic data.

This project adopts GPU parallel computing technology and efficient boundaries storage strategy to perform elastic FWI in time domain.  From the test results, it is obvious that the proposed schemes not only greatly improve the computational efficiency of FWI, but greatly reduce the amount of data storage. Another characteristic of time domain FWI is that it’s easy to fall into local minimum, this project performs multi-scale FWI to avoid this problem. Based on the 2D Overthrust model, the degree of dependence on initial model of multi-scale FWI is reduced. 

## Major features
- using CUDA accelerating elastic time domain fdtd.
- using efficient boundaries storage strategy to reduce the amount of data storage.
- Multiscale FWI to avoid the local minima.
- 2D Overthrust model  is used to verify the effectiveness of FWI.

## What's new

## User Guides


## TODO
- source encoding FWI
- more structured program code

## Citation
If you find this project useful in your research, please consider cite:
```
@article{2012A,
  title={A new scheme for elastic full waveform inversion based on velocity-stress wave equations in time domain},
  author={ Wang, Jie  and  Zhou, Hui  and  Tian, Yukun  and  Zhang, Hongjing },
  year={2012},
}
```


## License
This project is released under the [MIT License](LICENSE).

