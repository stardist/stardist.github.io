
# Frequently Asked Questions (FAQ)

<!--
* [Data](#data)
    * [Will it work for my data/application?](#will-it-work-for-my-data-application)
        * [How do I know if my objects of interest are (sufficiently) star-convex, i.e. is StarDist a good choice for my data?](#how-do-i-know-if-my-objects-of-interest-are-sufficiently-star-convex-i-e-is-stardist-a-good-choice-for-my-data)
        * [Other stains/markers with different appearance, quality, or inhomogeneity?](#other-stains-markers-with-different-appearance-quality-or-inhomogeneity)
        * [Can other objects besides round nuclei be segmented (e.g. multi-lobe nuclei, granules, bacteria)?](#can-other-objects-besides-round-nuclei-be-segmented-e-g-multi-lobe-nuclei-granules-bacteria)
        * [With multiple nucleus types, is it possible to only segment some or classify in addition to segmentation?](#with-multiple-nucleus-types-is-it-possible-to-only-segment-some-or-classify-in-addition-to-segmentation)
        * [Use for cell counting or centroid localization?](#use-for-cell-counting-or-centroid-localization)
    * [Data format/pre-processing](#data-format-pre-processing)
        * [Do I need to pre-process my images (e.g. background subtraction, filtering)?](#do-i-need-to-pre-process-my-images-e-g-background-subtraction-filtering)
        * [Is it advantageous to preprocess 3D stacks to adjust the axial resolution?](#is-it-advantageous-to-preprocess-3d-stacks-to-adjust-the-axial-resolution)
        * [Is a specific image format, size, or normalization required?](#is-a-specific-image-format-size-or-normalization-required)
        * [Are multi-channel images supported?](#are-multi-channel-images-supported)
* [Labeling/annotation](#labeling-annotation)
    * [How to label](#how-to-label)
        * [Should I annotate a few entire raw images/stacks, or is it better to annotate several smaller image crops?](#should-i-annotate-a-few-entire-raw-images-stacks-or-is-it-better-to-annotate-several-smaller-image-crops)
        * [Which size should the training images be?](#which-size-should-the-training-images-be)
        * [Is there an upper size limit for objects to be well segmented?](#is-there-an-upper-size-limit-for-objects-to-be-well-segmented)
        * [Do I have to annotate all nuclei (objects) in a training image? What about those that are only partially visible? What about other objects not of interest?](#do-i-have-to-annotate-all-nuclei-objects-in-a-training-image-what-about-those-that-are-only-partially-visible-what-about-other-objects-not-of-interest)
        * [Is it better to annotate images from scratch or to bootstrap/curate imperfect annotations (e.g. from another method)? Is training sensitive to annotation mistakes?](#is-it-better-to-annotate-images-from-scratch-or-to-bootstrap-curate-imperfect-annotations-e-g-from-another-method-is-training-sensitive-to-annotation-mistakes)
        * [How many images or nucleus (object) instances do I have to annotate for good results?](#how-many-images-or-nucleus-object-instances-do-i-have-to-annotate-for-good-results)
    * [Software/format for labeling](#software-format-for-labeling)
        * [In which format do I need to save my image annotations?](#in-which-format-do-i-need-to-save-my-image-annotations)
        * [Which software do you recommend to annotate 2D and 3D images?](#which-software-do-you-recommend-to-annotate-2d-and-3d-images)
        * [I've annotated my images in software X, how do I export the annotations as label images?](#i-ve-annotated-my-images-in-software-x-how-do-i-export-the-annotations-as-label-images)
* [Using pretrained models](#using-pretrained-models)
    * [How do I know if a pretrained (or any) model is suitable/good enough for my data?](#how-do-i-know-if-a-pretrained-or-any-model-is-suitable-good-enough-for-my-data)
    * [Do I need to rescale my images? How do I know which pixel resolution is required?](#do-i-need-to-rescale-my-images-how-do-i-know-which-pixel-resolution-is-required)
    * [Is there a pretrained model for 3D, or do you plan to release one?](#is-there-a-pretrained-model-for-3d-or-do-you-plan-to-release-one)
    * [Do you have plans to release other pretrained models?](#do-you-have-plans-to-release-other-pretrained-models)
* [Speed/Hardware/GPU](#speed-hardware-gpu)
    * [How can I speed up the prediction? Is it possible to predict on very large images/stacks?](#how-can-i-speed-up-the-prediction-is-it-possible-to-predict-on-very-large-images-stacks)
    * [What hardware do you recommend?](#what-hardware-do-you-recommend)
* [Method/technical](#method-technical)
    * [What are the probability and overlap/NMS thresholds? How do I select good values?](#what-are-the-probability-and-overlap-nms-thresholds-how-do-i-select-good-values)
    * [How does it work under the hood? I want to know technical details?](#how-does-it-work-under-the-hood-i-want-to-know-technical-details)
    * [Is a trained model sensitive to changes in image intensity or object size (as compared to the training images)?](#is-a-trained-model-sensitive-to-changes-in-image-intensity-or-object-size-as-compared-to-the-training-images)
    * [Do you support or recommend "transfer learning"?](#do-you-support-or-recommend-transfer-learning)
* [Postprocessing/quantification](#postprocessing-quantification)
    * [Is it possible to "refine" the shape of the predicted objects (e.g. for not fully star-convex objects)?](#is-it-possible-to-refine-the-shape-of-the-predicted-objects-e-g-for-not-fully-star-convex-objects)
    * [How do I evaluate the quality of the predicted results of a model?](#how-do-i-evaluate-the-quality-of-the-predicted-results-of-a-model)
    * [How can I perform measurements of the predicted objects in software X?](#how-can-i-perform-measurements-of-the-predicted-objects-in-software-x)
    * [How can I import the predicted results into software X?](#how-can-i-import-the-predicted-results-into-software-x)
* [Fiji/ImageJ](#fiji-imagej)
    * [After training in Python, how do I export a model to be used in Fiji? Do I have to be careful with the version of TensorFlow?](#after-training-in-python-how-do-i-export-a-model-to-be-used-in-fiji-do-i-have-to-be-careful-with-the-version-of-tensorflow)
    * [Can it be used in DeepImageJ?](#can-it-be-used-in-deepimagej)
    * [The Fiji plugin currently only supports 2D images. Is 3D support planned?](#the-fiji-plugin-currently-only-supports-2d-images-is-3d-support-planned)
    * [Are there differences between the Python and Fiji versions?](#are-there-differences-between-the-python-and-fiji-versions)
 -->


## Data

### Will it work for my data/application?


#### How do I know if my objects of interest are (sufficiently) star-convex, i.e. is StarDist a good choice for my data?

In a nutshell, most blob-like object shapes are star-convex (see [Wikipedia article](https://en.wikipedia.org/wiki/Star-shaped_polygon)). If you have labeled images, you can load your data in our [example notebooks](https://github.com/stardist/stardist/tree/master/examples) and see how well it can be reconstructed with a star-convex polygon/polyhedron representation. An average reconstruction IoU score (mean intersection of union score) of 0.8 or higher could be generally considered good enough.


#### Other stains/markers with different appearance, quality, or inhomogeneity?

Please first [verify that the shapes of your objects are star-convex](#how-do-i-know-if-my-objects-of-interest-are-sufficiently-star-convex-i-e-is-stardist-a-good-choice-for-my-data), i.e. blob-like. Examples of objects (segmentable by StarDist) include cells in brightfield images and stained structures in fluorescence or histology images. Where stains are used, an object can have its whole area stained, just its boundary stained, or be negatively stained (i.e. it is dark compared to other regions of the image). Next, please [check if one of the pretrained models works for your data](#how-do-i-know-if-a-pretrained-or-any-model-is-suitable-good-enough-for-my-data).

If your data is suitable for StarDist, but there is no pretrained model available, you need to train your own model. To that end, you need [labeled images](#labeling-annotation) before you can train your model (you can use the provided [example notebooks](https://github.com/stardist/stardist/tree/master/examples) where you replace the example data with your own).


#### Can other objects besides round nuclei be segmented (e.g. multi-lobe nuclei, granules, bacteria)?

The short answer is that StarDist should work well for segmenting all kinds of blob-like objects [with a star-convex shape](#how-do-i-know-if-my-objects-of-interest-are-sufficiently-star-convex-i-e-is-stardist-a-good-choice-for-my-data). However, it typically performs quite a bit better for roundish shapes compared to strongly elongated ones. For the latter, you often need to increase the number of rays to get decent results.


#### With multiple nucleus types, is it possible to only segment some or classify in addition to segmentation?

If there are multiple object/cell types in your image and you only want to segment some of them, you have several options. First, you can annotate only the object type(s) of interest in your training data, implicitly telling StarDist to consider everything else as background. While this can work, it might make it more difficult for StarDist to reliably distinguish between objects and background, especially if the visual differences between object types are subtle. Alternatively, you can annotate all objects in the training data, such that StarDist will learn to segment objects of all types. In a second step, you would have to filter out all objects of those types you are not interested in. This can either be done manually or with a different classification model.

Ideally, StarDist could additionally classify all objects while segmenting them. Although this is currently not possible, we might add this feature in a future version.


#### Use for cell counting or centroid localization?

If you just want to count or localize the centroids of cells, it might be a bit overkill to use StarDist (although trying one of the [pretrained models](https://github.com/stardist/stardist#pretrained-models-for-2d) is always a good idea). Dedicated cell counting and centroid localization approaches do exist, and they often need weaker forms of labeling, such as cell counts per training image or point annotations for cell centroids. However, if such centroid localization methods yield suboptimal results (e.g. in the case of very densely packed cells/nuclei) it might be worth to spend the extra annotation effort and train a dedicated StarDist model.


### Data format/pre-processing

#### Do I need to pre-process my images (e.g. background subtraction, filtering)?

In general, special pre-processing of images (such as background subtraction, denoising, etc) is not necessary. However it is reasonable to scale your input images such that the overall size of objects (in pixels) is similar to the size of objects used during training. If you have trained your own model, that means to always ensure that new images have roughly the same pixel size as the training images. This will make it much easier for StarDist to learn and might also avoid erroneous predictions of objects that are either too small or too large.

If you are using a [pre-trained model](https://github.com/stardist/stardist#pretrained-models-for-2d), it is important to know what kind of images it was trained with to understand if your image data is similar enough. In some cases, you can pre-process your images to make them suitable for a pre-trained model (e.g. up/downscaling of the image).


#### Is it advantageous to preprocess 3D stacks to adjust the axial resolution?

If your images contain only very few (<10) axial planes, you might consider doing a 2D segmentation from a maximum intensity projection (MIP) of the 3D stack. But the MIP should only be one or two cell layers thick. If you can't individualize cells by eye, there is little hope that StarDist will get it right.

If you need 3D segmentations, StarDist 3D does support anisotropic data (e.g. a 5x larger axial vs lateral pixel size should not be a problem). However, we sometimes found it advantageous to upscale the axial resolution to make objects appear more isotropic in the images. Hence, first try it directly with the anisotropic data and only if that doesn't lead to good results you could upscale the data isotropically. Note, that as one is not interested in restoring the image intensity signal but rather only segmenting the objects, it most likely would not make sense to use [Isotropic CARE](http://csbdeep.bioimagecomputing.com/examples/isotropic_reconstruction/).


#### Is a specific image format, size, or normalization required?

StarDist is in general not limited to images of specific formats, bit-depths, or sizes. Any input image however needs to be normalized to floating point values roughly in the range 0..1 before network prediction. Our [example notebooks](https://github.com/stardist/stardist/tree/master/examples) demonstrate how this normalization is done in Python, and our [Fiji plugin](https://imagej.net/StarDist) does this by default.

StarDist can be trained and predict on images with arbitrary spatial dimensions, but once a model is trained it is limited to its specific number of [input channels](#are-multi-channel-images-supported) (e.g. one cannot use a model trained for 2D RGB images on 2D single channel images).

StarDist does not put any constraints on the specific size of the input image: all padding and cropping necessary for the actual neural network is automatically handled for you. Also note that StarDist can do tiled prediction of large images in case of limited GPU memory.


#### Are multi-channel images supported?

A StarDist model is always trained to work for images with specific input channels in a given order. On one hand, that means you can train your own model with any number of input channels that you think might be helpful to accurately segment your images. On the other hand, these channels have to be always present in images that you want to segment using this model. Note that this also applies to the [pretrained models that we provide](#using-pretrained-models), which expect images with specific input channels.

If images have additional channels or channels in a different order than expected by a trained StarDist model, you first need to re-arrange them. For example, you may need to split the image channels and select the appropriate channel image (e.g. DAPI) before you can apply our  pretrained model for fluorescent nuclei in Fiji. You can then use the resulting segmentation to perform measurements in the other channels.

#### What if my training dataset does not fit into (CPU) memory?

If you cannot load your full training data into CPU memory (e.g. when using many large annotated 3D volumes), you can do the following:

* Use [keras Sequences](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence) to lazily load your images and masks
* Disable sample caching in the config `config = Config3D(..., train_sample_cache = False)`
* Use `model.train(X,Y,...)` with `X` and `Y` now being keras Sequence objects

This should lead to an almost constant memory footprint during training.


## Labeling/annotation

### How to label

#### Should I annotate a few entire raw images/stacks, or is it better to annotate several smaller image crops?

In general, it is better to annotate several image crops instead of entire (big) images or stacks. It is important that the content within annotated training images is representative of the content within images that you want to predict on later, after the model has been trained. In other words, the training data should cover the full range of variability that you expect in your (future) data.

#### Which size should the training images be?

As [mentioned earlier](#should-i-annotate-a-few-entire-raw-images-stacks-or-is-it-better-to-annotate-several-smaller-image-crops), it is generally better to annotate a variety of image crops as your training data. However, those crops must be big enough to contain entire fully visible objects and provide some context around them. Also make sure that not too many of the annotated objects are touching the border (it's fine if some do, but it should not be the majority). Example: if you have small cells with a diameter of 20 pixels, it might be sufficient to have annotated images of size 160x160, whereas if your objects have a diameter of 80 pixels, you would need to use larger annotated images e.g. of size 512x512. 

The "patch size" is an important parameter for training StarDist, and the size of images used for training affects what an appropriate value for the patch size should be (to maintain compatibility with the neural network architecture). For example, the patch size used for training StarDist must be smaller or equal than the size of the smallest annotated training image. To be on the safe side, ensure that the patch size is divisible by 16 along all dimensions. For example, you can annotate image crops of 300x300 pixels and then use a patch size of 256x256 pixels for training.


#### Is there an upper size limit for objects to be well segmented?

The maximal size of objects that can be well segmented depends on the receptive field of the neural network used inside a StarDist model.

For the default StarDist 2D network configuration, this is roughly 90 pixels. If your objects are larger than this and the segmentation results indicate over-segmentation, you can either a) downscale your input images such that the object size becomes smaller, or b) increase the receptive field of a StarDist model by changing the *grid* parameter in the model configuration (e.g. setting `grid=(2,2)` will roughly double the receptive field). Grid values of 4 and even 8 do make sense for images with a large minimum object size, e.g. 5x the size of the grid value.

This is similar for StarDist 3D, although the receptive field for the default network configuration is only roughly 35 pixels. Besides downscaling your input images, you can also change the grid parameter as mentioned above, but do not increase it for Z if you have strongly anisotropic images with relatively few axial planes, e.g. use `grid=(1,2,2)`. Furthermore, you can also slightly increase the receptive field by changing the *backbone* in the configuration to a U-Net, i.e. by using `backbone='unet'`.


#### Do I have to annotate all nuclei (objects) in a training image? What about those that are only partially visible? What about other objects not of interest?

Sparse labelling is not supported at this point, i.e. you must label all the objects in your chosen training images, even if they are only partially visible. If you don't do this, the trained model can be confused as to which pixels belong to objects and which belong to the background. As a consequence, this might result in many objects being missed during prediction.

If there are any other objects or structures present in the image, which are not of interest, there are two options. First, annotate them too for training and then filter them out later from the predicted objects. (In the future, we *might* add an option to additionally classify different objects types, making this easier.) I would recommend this in many cases, especially when the objects are also star-convex or look very similar to the objects of interest. Second, leave unwanted objects or structures out of the annotation if they can easily be distinguished from the objects of interest. If in doubt, try both strategies.


#### Is it better to annotate images from scratch or to bootstrap/curate imperfect annotations (e.g. from another method)? Is training sensitive to annotation mistakes?

In practice, you probably would like to use the labeling approach that requires the least amount of manual annotation/curation work. It depends on your data whether this is labeling from scratch or curating an imperfect automatic labeling.

Annotating images from scratch is often easier because it doesn't involve obtaining predictions and curating them. It can be a good strategy if the task is not too difficult.

If you already have an instance segmentation method with decent results, you can try training StarDist by using its predictions as ground truth. As long as there are no *systematic* mistakes in the ground truth, we have observed that training can still be successful. Especially when the segmentation task is more difficult (e.g. noisy images and/or strong appearance variations), it often makes sense to train an initial model, curate its predictions and add them to the training data.


#### How many images or nucleus (object) instances do I have to annotate for good results?

This is very difficult to answer in general, since it really depends on your specific data. The more variability is in your data (object shapes and packing, background, noise, signal variation, aberrations), the more training data (in form of a wide range of examples) is necessary so that the network can learn to perform accurate predictions.

We have often seen good results from as few as 5-10 image crops each having 10-20 annotated objects (in 2D), but your mileage may vary substantially. You can always start with a small training dataset, inspect/curate the results and iterate. 

Furthermore, one can/should always use *data augmentation* to artificially inflate the training data by adding training images with *plausible* appearance variations. What plausible means depends on the data at hand, but some operations (random flips and rotations, intensity shifts) can be used in most cases and are demonstrated e.g. in the training [example notebook](https://github.com/stardist/stardist/tree/master/examples). 


### Software/format for labeling

#### In which format do I need to save my image annotations?

The image annotations (also known as *label images* or *label masks*) should be integer-valued (e.g. 8-bit, 16-bit, 32-bit) TIFF files where all background pixels have value 0 and each object instance is represented by an area/volume filled with a unique integer value. It does not matter what the values are and they do not need to be consecutive. Please note that a foreground/background segmentation mask, where all object instances are denoted by the same value, is not sufficient for StarDist training.

Note that for visualization purposes, label images are often displayed with each object instance in a different color (to tell them apart) on a black background; this is the result of applying a look-up table (e.g. *Glasbey on dark* in Fiji). As mentioned above, the label masks for StarDist must be integer-valued TIFF files and not RGB files, i.e. the specific color does not matter.


#### Which software do you recommend to annotate 2D and 3D images?

In 2D, there are several options, among them being [Fiji](http://fiji.sc/), [QuPath](https://qupath.github.io), or [Labkit](https://imagej.net/Labkit). Although each of these provide decent annotation tools, we currently recommend using Labkit for its easy label export. Please read [here](https://github.com/stardist/stardist#annotating-images) for more detailed instructions how to use Labkit to generate annotations.  

In 3D, there are fewer options: [Labkit](https://github.com/maarzt/imglib2-labkit) and [Paintera](https://github.com/saalfeldlab/paintera) (the latter being very sophisticated but having a steeper learning curve).


#### I've annotated my images in software X, how do I export the annotations as label images?

Here is some advice for exporting annotations to a label image from different tools (see [here](#which-software-do-you-recommend-to-annotate-2d-and-3d-images) for a list of recommended tools). 

- Fiji: Use [this script](https://gist.github.com/maweigert/9f2684f36d3272786461a0c18d4ea176) to convert annotations to label images.
- Labkit: Please read [this](https://github.com/stardist/stardist#annotating-images).
- QuPath: See [this post](https://forum.image.sc/t/export-qupath-annotations-for-stardist-training/37391) to get started.


## Using pretrained models

#### How do I know if a pretrained (or any) model is suitable/good enough for my data?

First, you can take a look at the existing pretrained models and inspect the images they were trained on, to get an idea if one of them might be suitable for your data. At the moment, you can find an overview of pretrained models [here](https://github.com/stardist/stardist#pretrained-models-for-2d) and [here](https://imagej.net/StarDist#Plugin), including links to the training datasets. Furthermore, our [example notebooks](https://github.com/stardist/stardist/tree/master/examples) also demonstrate how to show a list of the available pretrained models.

If you found a promising pretrained model for your data, it is probably easiest to quickly try it out with our [Fiji plugin](https://imagej.net/StarDist) and manually inspect if the results are plausible. If that's the case, you may also want to [quantitatively evaluate the results](#how-do-i-evaluate-the-quality-of-the-predicted-results-of-a-model).


#### Do I need to rescale my images? How do I know which pixel resolution is required?

Besides being rather robust to intensity changes, our pretrained models are able to segment objects with a fair range of sizes. Please take a look at the respective training datasets to get an idea of the object size variations that the model should be able handle. In the future, we might provide additional metadata for each pretrained model to help you with that. Also please have a look [at this related question](#is-there-an-upper-size-limit-for-objects-to-be-well-segmented).

If your images contain relatively large objects and you observe lots of over-segmentation mistakes (i.e. several smaller objects predicted instead of an expected large one), you should try to reduce the pixel resolution of the image before applying StarDist.


#### Is there a pretrained model for 3D, or do you plan to release one?

Unfortunately not, but we would like to provide one at some point. A major issue is the lack of available training data.


#### Do you have plans to release other pretrained models?

There are no immediate plans at the moment, but we can relatively easily be persuaded to add new ones given a common use case and the availability of suitable training data.


## Speed/Hardware/GPU

#### How can I speed up the prediction? Is it possible to predict on very large images/stacks?

StarDist prediction consists of two phases:

1. Neural network prediction based on a normalized input image. This can optionally be GPU-accelerated (in both Python and Fiji) if TensorFlow with the necessary dependencies is installed. (GPU acceleration is very much recommended for 3D images and large 2D images.)

2. Post-processing of the neural network output, which involves a non-maximum suppression step (using the provided probability and overlap thresholds) to prune redundant object instances. This step does not use the GPU (and cannot reasonably be changed to do so), but will take advantage of all available CPU cores, i.e. can be substantially faster on more powerful multi-core CPUs. StarDist was properly installed with multi-core (OpenMP) support if it is running on several CPU cores while predicting.

In order to handle large images (or stacks) that cannot be processed all at once in step 1, there is an option to internally process the input image in separate overlapping tiles. To that end, you can specify the number of tiles in both Python (parameter `n_tiles` of `model.predict`) and Fiji. This is especially necessary because GPUs often have limited memory that does not permit to process large images directly.

Step 2 is currently processed for the entire image, which can be a computational bottleneck for large images. To alleviate this issue, we are currently working on another option that will allow us to also perform this step independently for regions of the image.
Note that we also consider supporting cases where the input image (and resulting prediction) are too large to fit in host memory, i.e. cannot be loaded all at once.


#### What hardware do you recommend?

If you occasionally want to segment 2D images of moderate size (e.g. 1024x1024 pixels), you do not need special hardware – a typical laptop will be enough. However, as [mentioned before](#how-can-i-speed-up-the-prediction-is-it-possible-to-predict-on-very-large-images-stacks), both a more powerful multi-core CPU and a recent GPU can substantially speed up prediction with StarDist. Furthermore, training your own model without a GPU is not recommended at all, as this can be *very* slow, especially for 3D images. Of course, if you intend to use StarDist for large images or stacks, you will need a sufficient amount of RAM and storage.

Regarding the choice of GPU, it (currently) has to be a [CUDA](https://en.wikipedia.org/wiki/CUDA)-compatible GPU from Nvidia. There are many options to choose from, which do change all the time. A very important factor besides speed is the amount of GPU memory, which should be 8 GB or more when training StarDist 3D models. It is much less important for 2D training and prediction (in both 2D and 3D). We personally use rather high-end (but now 3+ years old) GPUs (1080, Titan X Maxwell, Titan X Pascal).


## Method/technical

#### What are the probability and overlap/NMS thresholds? How do I select good values?

StarDist internally uses a neural network to predict two separate quantities per pixel, 1) an object probability and 2) several distances to the object boundary that the pixel belongs to. Only pixels with an object probability above a chosen *probability threshold* are allowed to "vote" for an object candidate (i.e. a star-convex polygon defined via the predicted distances). Note that many pixels will vote for similar object candidates, since they belong to the same object. Hence, after all object candidates have been collected, a non-maximum suppression (NMS) step is used to prune all the redundant objects, such that (ideally) only one object is retained for every true object in the image. To that end, we need to define which object candidates likely represent the same object in the image. We use a typical approach by defining object similarity in terms of overlap, i.e. two objects are considered equal if their (normalized) intersection area/volume exceeds an *overlap/NMS threshold*.

At the end of our [training notebooks](https://github.com/stardist/stardist/tree/master/examples), we automatically optimize both thresholds based on your validation data, such that they should yield good results in many cases. However, both thresholds can be adjusted to your specific application. Higher values of the probability threshold can yield fewer segmented objects, but will likely avoid false positives. Higher values of the overlap threshold will allow segmented objects to overlap more. If your objects should never overlap, you may set the overlap threshold close to 0. 


#### How does it work under the hood? I want to know technical details.

Please see the [high-level overview](#what-are-the-probability-and-overlap-nms-thresholds-how-do-i-select-good-values) above. If you want to know more, please have a look at [our documentation](https://github.com/stardist/stardist) and the papers:

- Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.  
[*Cell Detection with Star-convex Polygons*](https://arxiv.org/abs/1806.03535).  
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

- Martin Weigert, Uwe Schmidt, Robert Haase, Ko Sugawara, and Gene Myers.  
[*Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy*](http://openaccess.thecvf.com/content_WACV_2020/papers/Weigert_Star-convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.pdf).  
The IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass Village, Colorado, March 2020


#### Is a trained model sensitive to changes in image intensity or object size (as compared to the training images)?

A trained model will typically only work well for images that are similar to those that the model was trained on. However, one can use *data augmentation* during training to synthetically vary image intensities and object sizes. As a result, the trained model will be robust towards these variations, since it was trained to expect these. Also see this [previous question](#how-do-i-know-if-a-pretrained-or-any-model-is-suitable-good-enough-for-my-data).


#### Do you support or recommend "transfer learning"?

While [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) is promising in case of limited annotated training data, we currently do not support it because we haven't investigated how and when to use it. (However, it is possible with the current code, but simply not documented.)

Furthermore, we have made the observation that training (from scratch) using a combination of a small custom dataset together with an existing bigger (and somewhat similar) dataset can lead to better results than just training with the custom data.


## Postprocessing/quantification

#### Is it possible to "refine" the shape of the predicted objects (e.g. for not fully star-convex objects)?

It is possible, but not supported in our software at the moment. We are looking into this but can't promise if and when a solution will be available. However, some people have already used StarDist to generate high-quality *seeds* and then used other seed-based methods (e.g. watershed) to obtain instance segmentations that are not restricted to star-convex object shapes.


#### How do I evaluate the quality of the predicted results of a model?

Ultimately, this depends on your application, i.e. what you want to do with the segmentation results (e.g. counting, intensity measurements, tracking). Hence, we consider a typical evaluation approach here, which we also carry out at the end of our [training notebooks](https://github.com/stardist/stardist/tree/master/examples).

The detection/segmentation performance can be quantitatively evaluated by considering objects in the ground truth to be correctly matched if there are predicted objects with overlap ([intersection over union (IoU)](https://en.wikipedia.org/wiki/Jaccard_index)) beyond a chosen IoU threshold (value between 0 and 1).
The obtained matching statistics (accuracy, recall, precision, etc.) can be quite informative for the model's performance (see [sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) for further details).
The IoU threshold can be between 0 (even slightly overlapping objects count as correctly predicted) and 1 (only pixel-perfectly overlapping objects count) and which threshold to use depends on the needed segmentation precision/application.


#### How can I perform measurements of the predicted objects in software X?

The output of StarDist is a label image and/or a list of (polygon/polyhedron) ROIs, one for each object. These can be used for quantification, for example:

* Python: Based on the label image, the function [regionprops](https://scikit-image.org/docs/0.17.x/api/skimage.measure.html#skimage.measure.regionprops) (or [regionprops_table](https://scikit-image.org/docs/0.17.x/api/skimage.measure.html#skimage.measure.regionprops_table)) from [scikit-image](https://scikit-image.org) offers many different measurements for each object instance. Note that you can also [export 2D predictions as ImageJ ROIs](https://github.com/stardist/stardist/blob/master/examples/other2D/export_imagej_rois.ipynb).

* Fiji/ImageJ: The *ROI Manager* can be used to measure many different properties (which can be chosen via *Analyze > Set Measurements...*)


#### How can I import the predicted results into software X?

As [mentioned above](#how-can-i-perform-measurements-of-the-predicted-objects-in-software-x), StarDist can output its predictions as label images, or lists of polygon/polyhedron coordinates. Label images are quite universal and can be imported in many different software packages. In Python, the 2D polygon coordinates can also be [exported as ImageJ ROIs](https://github.com/stardist/stardist/blob/master/examples/other2D/export_imagej_rois.ipynb), or be serialized to different formats via [Shapely](https://github.com/Toblerity/Shapely).


## Fiji/ImageJ

#### After training in Python, how do I export a model to be used in Fiji? Do I have to be careful with the version of TensorFlow?

After training your StarDist model in Python, you can export it to be used in [Fiji](https://imagej.net/StarDist) (or [QuPath](https://qupath.readthedocs.io/en/latest/docs/advanced/stardist.html)) by calling `model.export_TF()`. This will create a ZIP file that contains the trained model in the correct format.

It is important that the version of TensorFlow (a neural network library that StarDist depends on) used in Fiji (or QuPath) is the same or newer as in Python. You can find out which version is used in Python via `import tensorflow; print(tensorflow.__version__)`. In Fiji, you can manage your version of TensorFlow via *Edit > Options > TensorFlow...*. Note that this also applies to our pretrained models, which currently require TensorFlow 1.12.0 or newer.

<del>Note that StarDist currently *only* supports TensorFlow 1.x, i.e. do not upgrade or install a recent 2.x version.</del>
Starting with version 0.6.0, StarDist for Python does work with either TensorFlow 1 or 2. Furthermore, when using TensorFlow 2, it appears that an exported model will work in Fiji with TensorFlow 1.14.0.


#### Can it be used in DeepImageJ?

We recommend using [our plugin](https://imagej.net/StarDist) when using StarDist in Fiji, because it bundles all the necessary steps.

However, if you are an advanced user and want to use [DeepImageJ](https://deepimagej.github.io/deepimagej/), you should be able to do so with a [pretrained](#using-pretrained-models) or [exported model](#after-training-in-python-how-do-i-export-a-model-to-be-used-in-fiji-do-i-have-to-be-careful-with-the-version-of-tensorflow). However, this will only perform the neural network prediction and not the necessary [non-maximum suppression (NMS)](#what-are-the-probability-and-overlap-nms-thresholds-how-do-i-select-good-values) step. You can call just the NMS step from our plugin (*Plugins > StarDist > Other > StarDist 2D NMS (postprocessing only)*) though. Note that we haven't tested this workflow, but it should work in principle.


#### The Fiji plugin currently only supports 2D images. Is 3D support planned?

Yes, we also want to support 3D in our [Fiji plugin](https://imagej.net/StarDist). However, there are some issues that we need to solve first, especially related to deployment (reliance on C++ code that isn't easily portable to Java).


#### Are there differences between the Python and Fiji versions?

Regarding the prediction results of the neural network, they should be identical or only have negligible differences. However, our [Python package](https://github.com/stardist/stardist) is the reference implementation with the most features, some of which are missing in Fiji.

For example, besides lacking model training and 3D support, the Fiji plugin currently does not offer different normalization options for multi-channel images or quantitative evaluation of prediction results.
