# CameraCalibration

This one is the implemenation if the thesis "A flexible new technique for camera calibration". 

The sample image is like below : 

<img src="https://github.com/WD4715/CameraCalibration/assets/117700793/bb873e98-ca2d-47fc-921e-75b1565da0d0" width="400" height="300">

And Using Camera calibration, we can get the output(Camera intrinsic and extrinsic matrix). Using this output, we can reproject the chessboard conner. and the output will be the same below:

<img src="https://github.com/WD4715/CameraCalibration/assets/117700793/9ffa3d79-6f14-4c05-b098-10a2f4e7740c" width="400" height="300">

The run code is simple : 

```
$ python main.py 
```
