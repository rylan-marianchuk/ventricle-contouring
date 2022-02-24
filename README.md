# VentricleContouring

*Author: Rylan Marianchuk*
*February 2022*

Python >= 3.7 required.
Package dependencies:
```
numpy
```
Optional package dependencies for output/debug visualization and image viewing:
```
plotly
imageio
```

Given two binary masks as ndarrays: 
* ```lumen_mask``` places 1 throughout the ventricle volume
* ```myo_mask``` places 1 on the myocardium (lining the ventricle)

![Myocardiam mask (left), Solid lumen mask (right)](img/readme/myo-lumen.png)

*Myocardiam mask (left), Solid lumen mask (right)*

The endocardium and epicardium contours, in addition to the apex, can be obtained from these masks by calling the ``MaskToContour()`` object:
```
GetContour = MaskToContour()
endo, epi, apex = GetContour(solid_mask, myo_mask)
```

See the docstring in ``__init__()`` and ``__call__()`` for specfic shapes and dtypes and optional parameters to invoke.
See ```main.py``` for code that reads in masks and executes the transform on a .png image.

Constraints:
* Contours are equi-distant
* Endo and Epi countours shall never cross
* Any endo point will always be at least 1 unit away from any epi point
* Invariant to ventricle rotation

*Example output of a contour density of 100 points, overlayed on MR image using the default parameters:*

```
GetContour = MaskToContour(debug=True, dPhi=0.01, dR=0.5, contourDensity=100)
endo, epi, apex = GetContour(solid_mask, myo_mask)
```
<img alt="Example output overlayed on MR image" height="500" src="img/readme/result-vis.png" width="500"/>
