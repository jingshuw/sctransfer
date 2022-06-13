# The supporting Python package of SAVER-X

This package is designed to be used by the R code of [SAVER-X](https://github.com/jingshuw/SAVERX).
It contains simplified code from the python [dca](http://github.com/theislab/dca) package and the new Python code for transfer learning.

One can also use this Python package for pre-training using public data. Instructions on pre-training will come out soon.

## Installation:

```
pip install sctransfer
```
The package only supports Python (>=3.5). 

Update: the package now works with tensorflow 2 

Update: the package requires tensorflow version <=2.2.0 and keras version <=2.3.0.

Update: the package has been updated to work with the newer version of scanpy.

To test whether the installation is successful or not, you can try with the following code. The mtx file for testing can be downloaded [here](https://www.dropbox.com/s/qy2wp2i64jjtuti/shekhar_downsampled.mtx?dl=0) 
```
import sctransfer.api as api
api.autoencode(mtx_file = "shekhar_downsampled.mtx")
```
