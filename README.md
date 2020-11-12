# Twitter Flood Datasets

This repository contains two datasets of flood images from Twitter:
The *Harz17* dataset comprises images from tweets containing flood-related keywords during the occurrence of a flood in the Harz region in Germany in July 2017.
Similarly, the *Rhine18* dataset comprises images related to a flood of the river Rhine in January 2018.

Furthermore, we provide expert annotations for all images in the dataset, indicating whether the image is relevant for one of the following tasks:

- Determining whether a certain area is flooded.
- Deriving an estimation of the inundation depth from the image due to visual cues such as, for example, traffic signs or other structures with known height.
- Assessing the grade of water pollution by substances like oil, for instance?

Moreover, we provide two models based on ResNet-50 and VGG-16 for predicting each of these labels.
The final classification layer of the models has been trained on the independent [European Flood 2013 dataset][1], which comprises flood images from Wikimedia Commons.

The datasets, annotations, models, and experimental evaluation are described in the following paper:

> Björn Barz, Kai Schröter, Ann-Christin Kra, and Joachim Denzler.  
> ["Finding Relevant Flood Images on Twitter using Content-based Filters."][2]  
> ICPR 2020 Workshop on Machine Learning Advances Environmental Science.


## Annotations

The annotations are provided in the files [`harz17.json`](harz17.json) and [`rhine18.json`](rhine18.json) and look like this:

```json
{
    "DDyrSTxUQAAZM_A": {
        "TweetID": 881767997014065152,
        "URL": "http://pbs.twimg.com/media/DDyrSTxUQAAZM_A.jpg",
        "Timestamp": 1499064847,
        "RelFlooding": true,
        "RelDepth": true,
        "RelPollution": false
    },
    ...
}
```

The keys of the object contained in each file are unique identifiers of all images in the dataset, derived from their filenames.
For each image, we provide the ID of the tweet, the original URL of the image, the UNIX timestamp of the tweet, and the three binary class labels.


## Obtaining the Images

Due to Twitter's Developer Agreement and Policy, we are not allowed to re-distribute the actual images nor the contents of the tweets.
You may still access the full information of all tweets using the provided tweet ID.

The python script `download_images.py` can be used for downloading the actual images.
It will create two directories, `harz17` and `rhine18`, and download all images that are still available.


## Classification Models

We provide two [Keras][5] models for predicting the three class labels mentioned above based on the ResNet-50 and the VGG-16 architecture.
Due to their filesize, they have to be downloaded separately and placed in the `models` directory: [ResNet-50][3] (91 MB) | [VGG-16][4] (57 MB)

The models are pre-trained on ImageNet and the final classification layer has been initialized using an SVM.
Please refer to the abovementioned paper for details.

### Pre-processing

The models expect images in BGR color format with zero-centered color values using the following mean:

- Blue: 103.939
- Red: 116.779
- Green: 123.68

Furthermore, images should be resized so that they fully cover a bounding box of size 768x512 (for landscape images) or 512x768 (for portrait images), but are not larger than necessary.
After resizing, a central crop of that size should be extracted.
An example for how to implement this can be found in [`utils.py`](utils.py).

### Classification Thresholds

We obtained optimal F1-scores with the following classification thresholds:

|  Model / Class  | Flooding |  Depth  | Pollution |
|-----------------|---------:|--------:|----------:|
| [ResNet-50][3]  |  -0.0677 | -0.1800 |    0.0110 |
| [VGG-16][4]     |  -0.1450 | -0.2835 |   -0.1408 |

### Usage Examples

Examples for loading the datasets, using, and evaluating the models can be found in [`Flood-Classification.ipynb`](Flood-Classification.ipynb).

The following Python packages are required for running the notebook:

- numpy
- pandas
- matplotlib
- scikit-learn
- keras <= 2.2
- keras-preprocessing == 1.1
- tensorflow 1.x


[1]: https://github.com/cvjena/eu-flood-dataset
[2]: https://arxiv.org/pdf/2011.05756.pdf
[3]: https://github.com/cvjena/twitter-flood-dataset/releases/download/v1.0/flood_relevance.rn50.h5
[4]: https://github.com/cvjena/twitter-flood-dataset/releases/download/v1.0/flood_relevance.vgg16.h5
[5]: https://keras.io
