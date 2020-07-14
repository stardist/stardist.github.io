# Overview

The following figure illustrates the general approach for 2D images. The training data consists of corresponding pairs of input (i.e. raw) images and fully annotated label images (i.e. every pixel is labeled with a unique object id or 0 for background). 
A model is trained to densely predict the distances (r) to the object boundary along a fixed set of rays and object probabilities (d), which together produce an overcomplete set of candidate polygons for a given input image. The final result is obtained via non-maximum suppression (NMS) of these candidates.  

![](https://github.com/mpicbg-csbd/stardist/raw/master/images/overview_2d.png)

The approach for 3D volumes is similar to the one described for 2D, using pairs of input and fully annotated label volumes as training data.

![](https://github.com/mpicbg-csbd/stardist/raw/master/images/overview_3d.png)

## Webinar/Tutorial

If you want to know more about the concepts and practical applications of StarDist, please have a look at the following webinar that was given at NEUBIAS Academy @Home 2020:

<iframe width="560" height="315" src="https://www.youtube.com/embed/Amn_eHRGX5M" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<br>&nbsp;


# Installation

This package requires Python 3.6 (or newer).

1. Please first [install TensorFlow](https://www.tensorflow.org/install)
(either TensorFlow 1 or 2) by following the official instructions.
For [GPU support](https://www.tensorflow.org/install/gpu), it is very
important to install the specific versions of CUDA and cuDNN that are
compatible with the respective version of TensorFlow.

2. *StarDist* can then be installed with `pip`:

    `pip install stardist`

## Notes

- Depending on your Python installation, you may need to use `pip3` instead of `pip`.
- Since this package relies on a C++ extension, you could run into compilation problems (see [Troubleshooting](#troubleshooting) below). We currently do not provide pre-compiled binaries.
- StarDist uses the deep learning library [Keras](https://keras.io), which requires a suitable [backend](https://keras.io/backend/#keras-backends) (we currently only support [TensorFlow](http://www.tensorflow.org/)).
- *(Optional)* You need to install [gputools](https://github.com/maweigert/gputools) if you want to use OpenCL-based computations on the GPU to speed up training.
- *(Optional)* You might experience improved performance during training if you additionally install the [Multi-Label Anisotropic 3D Euclidean Distance Transform (MLAEDT-3D)](https://github.com/seung-lab/euclidean-distance-transform-3d).

## Troubleshooting

Installation requires Python 3.6 (or newer) and a working C++ compiler. We have only tested [GCC](http://gcc.gnu.org) (macOS, Linux), [Clang](https://clang.llvm.org) (macOS), and [Visual Studio](https://visualstudio.microsoft.com) (Windows 10). Please [open an issue](https://github.com/mpicbg-csbd/stardist/issues) if you have problems that are not resolved by the information below.

If available, the C++ code will make use of [OpenMP](https://en.wikipedia.org/wiki/OpenMP) to exploit multiple CPU cores for substantially reduced runtime on modern CPUs. This can be important to prevent slow model training.


### macOS
The default Apple C/C++ compiler (`clang`) does not come with OpenMP support and the package build will likely fail.
To properly build `stardist` you need to install a OpenMP-enabled GCC compiler, e.g. via [Homebrew](https://brew.sh) with `brew install gcc` (which will currently install `gcc-9`/`g++-9`). After that, you can build the package like this (adjust compiler names/paths as necessary):

    CC=gcc-9 CXX=g++-9 pip install stardist

If you use `conda` on macOS and after `import stardist` see errors similar to the following:

    Symbol not found: _GOMP_loop_nonmonotonic_dynamic_next

please see [this issue](https://github.com/mpicbg-csbd/stardist/issues/19#issuecomment-535610758) for a temporary workaround.  

### Windows
Please install the [Build Tools for Visual Studio 2019](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2019) from Microsoft to compile extensions for Python 3.6 and newer (see [this](https://wiki.python.org/moin/WindowsCompilers) for further information). During installation, make sure to select the *C++ build tools*. Note that the compiler comes with OpenMP support.


# Usage

We provide example workflows for 2D and 3D via Jupyter [notebooks](https://github.com/mpicbg-csbd/stardist/tree/master/examples) that illustrate how this package can be used.

![](https://github.com/mpicbg-csbd/stardist/raw/master/images/example_steps.png)

## Pretrained Models for 2D

Currently we provide some pretrained models in 2D that might already be suitable for your images:

<table border="1">
<thead>
<tr>
<th style="padding: 5px" align="center">Key (Name)</th>
<th style="padding: 5px" align="center">Modality (Staining)</th>
<th style="padding: 5px" align="center">Image Format</th>
<th style="padding: 5px" align="center">Example Image</th>
<th style="padding: 5px" align="center">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding: 5px" align="center"><code>2D_versatile_fluo</code><br/> <code>2D_paper_dsb2018</code></td>
<td style="padding: 5px" align="center">Fluorescence (nuclear marker)</td>
<td style="padding: 5px" align="center">2D single channel</td>
<td style="padding: 5px" align="center"><img src="https://github.com/mpicbg-csbd/stardist/raw/master/images/example_fluo.jpg" title="example image fluo" width="120px" align="center" style="max-width:100%;"></td>
<td style="padding: 5px" align="center"><em>Versatile (fluorescent nuclei)</em> and <em>DSB 2018 (from StarDist 2D paper)</em> that were both trained on a subset of the <a href="https://data.broadinstitute.org/bbbc/BBBC038/">DSB 2018 nuclei segmentation challenge dataset</a>.</td>
</tr>
<tr>
<td style="padding: 5px" align="center"><code>2D_versatile_he</code></td>
<td style="padding: 5px" align="center">Brightfield (H&amp;E)</td>
<td style="padding: 5px" align="center">2D RGB</td>
<td style="padding: 5px" align="center"><img src="https://github.com/mpicbg-csbd/stardist/raw/master/images/example_histo.jpg" title="example image histo" width="120px" align="center" style="max-width:100%;"></td>
<td style="padding: 5px" align="center"><em>Versatile (H&amp;E nuclei)</em> that was trained on images from the <a href="https://monuseg.grand-challenge.org/Data/">MoNuSeg 2018 training data</a> and the <a href="http://cancergenome.nih.gov/">TCGA archive</a>.</td>
</tr>
</tbody>
</table>
<br/>

<!--
| key | Modality (Staining) | Image format | Example Image    | Description  | 
| :-- | :-: | :-:| :-:| :-- |
| `2D_versatile_fluo` `2D_paper_dsb2018`| Fluorescence (nuclear marker) | 2D single channel| <img src="https://github.com/mpicbg-csbd/stardist/raw/master/images/example_fluo.jpg" title="example image fluo" width="120px" align="center">       | *Versatile (fluorescent nuclei)* and *DSB 2018 (from StarDist 2D paper)* that were both trained on a subset of the [ DSB 2018 nuclei segmentation challenge dataset](https://data.broadinstitute.org/bbbc/BBBC038/). | 
|`2D_versatile_he` | Brightfield (H&E) | 2D RGB  | <img src="https://github.com/mpicbg-csbd/stardist/raw/master/images/example_histo.jpg" title="example image histo" width="120px" align="center">       | *Versatile (H&E nuclei)* that was trained on images from the [MoNuSeg 2018 training data](https://monuseg.grand-challenge.org/Data/) and the [TCGA archive](http://cancergenome.nih.gov/). |
-->

You can access these pretrained models from `stardist.models.StarDist2D`

```python
from stardist.models import StarDist2D 

# prints a list of available models 
StarDist2D.from_pretrained() 

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')
```


## Annotating Images

To train a *StarDist* model you will need some ground-truth annotations: for every raw training image there has to be a corresponding label image where all pixels of a cell region are labeled with a distinct integer (and background pixels are labeled with 0). 
To create such annotations in 2D, there are several options, among them being [Fiji](http://fiji.sc/), [Labkit](https://imagej.net/Labkit), or [QuPath](https://qupath.github.io). In 3D, there are fewer options: [Labkit](https://github.com/maarzt/imglib2-labkit) and [Paintera](https://github.com/saalfeldlab/paintera) (the latter being very sophisticated but having a steeper learning curve). 

Although each of these provide decent annotation tools, we currently recommend using Labkit (for 2D or 3D images) or QuPath (for 2D):

### Annotating with LabKit (2D or 3D)

1. Install [Fiji](https://fiji.sc) and the [Labkit](https://imagej.net/Labkit) plugin
2. Open the (2D or 3D) image and start Labkit via `Plugins > Segmentation > Labkit`
3. Successively add a new label and annotate a single cell instance with the brush tool (always check the `override` option) until *all* cells are labeled
4. Export the label image via `Save Labeling...` and `File format > TIF Image` 

![](https://github.com/mpicbg-csbd/stardist/raw/master/images/labkit_2d_labkit.png)

Additional tips:

* The Labkit viewer uses [BigDataViewer](https://imagej.net/BigDataViewer) and its keybindings (e.g. <kbd>s</kbd> for contrast options, <kbd>CTRL</kbd>+<kbd>Shift</kbd>+<kbd>mouse-wheel</kbd> for zoom-in/out etc.)
* For 3D images (XYZ) it is best to first convert it to a (XYT) timeseries (via `Re-Order Hyperstack` and swapping `z` and `t`) and then use <kbd>[</kbd> and <kbd>]</kbd> in Labkit to walk through the slices.    

### Annotating with QuPath (2D)

1. Install [QuPath](https://qupath.github.io/)
2. Create a new project (`File -> Project...-> Create project`) and add your raw images 
3. Annotate nuclei/objects
4. Run [this script](https://raw.githubusercontent.com/mpicbg-csbd/stardist/master/extras/qupath_export_annotations.groovy) to export the annotations (save the script and drag it on QuPath. Then execute it with `Run for project`). The script will create a `ground_truth` folder within your QuPath project that includes both the `images` and `masks` subfolder that then can directly be used with *StarDist*.

To see how this could be done, have a look at the following [example QuPath project](https://raw.githubusercontent.com/mpicbg-csbd/stardist/master/extras/qupath_example_project.zip) (data courtesy of Romain Guiet, EPFL). 

![](https://github.com/mpicbg-csbd/stardist/raw/master/images/qupath.png)


# ImageJ/Fiji Plugin

We currently provide a ImageJ/Fiji plugin that can be used to run pretrained StarDist models on 2D or 2D+time images. Installation and usage instructions can be found at the [plugin page](https://imagej.net/StarDist).