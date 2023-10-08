# Semantic Segmentation of Bridge Structural Components

We use [Keras Tensorflow](https://www.tensorflow.org/api_docs/python/tf) to create the model. There is a script to also convert a model to the [ONNX](https://onnx.ai/) format.

## Prerequisites
`requirements.txt` contains the necessary packages.

The assumed directory structure used for training is:

```mermaid
flowchart TD
    A[data directory] --> B[train]
    A --> C[validation]
    A --> D[test]
    B --> E[images]
    B --> F[masks]
    C --> G[images]
    C --> H[masks]
    D --> I[images]
    D --> J[masks]
    F --> K[label_0]
    F --> L[label_1]
    F --> M[...]
    H --> N[label_0]
    H --> O[label_1]
    H --> P[...]
    J --> Q[label_0]
    J --> R[label_1]
    J --> S[...]
 ```

<a name="data_struct"> </a>
 If such folder structure does not exist, the script `split_dataset.py` will create it,
 assuming the structure is:
 ```mermaid
 flowchart TD
 A[data directory] --> B[images]
 A --> C[masks]
 C --> D[label_0]
 C --> E[label_1]
 C --> F[...]
 ```

## Usage

The code is divided in a number of modules.
1. `semantic_segm_pipeline.py` is the most loaded script of all. It trains and then tests a model, on either real or synthetic data. If a model is given as argument, along with the flaf `--finetune`, the existing model is finetuned. Many of the model specifications and the dataset to train on are given as arguments to the script.
2. `test_pipeline.py`. Takes as argument a model and a data directory (assuming [this](#data_struct) structure), and tests the model. This functionality is included by default in point 1., but exists also seperately to allow for more tests.
3. `split dataset.py`. Given a datapath with [this](#data_struct) structure, transforms it in the training appropriate (creates `train`, `validation` and `test` folders, and distributes the data with a 60-20-20 training-validation-testing split ratio.)
4. `tfKeras2onnx.py`. Takes a tensorflow keras model (SavedModel format), and transforms it into ONNX.
5. `test_onnx.py`. Useful for debugging, takes as input an ONNX model and tests it with one or more test images.
6. `model_complexity.py`. Auxiliary module, creating a csv file with number of parameters and inference time for all implemented models.
7. `extract_cvat_annotations.py`. Given an image set and an xml file exported from [CVAT annotation tool](https://cvat.org/), CVAT for images format (one xml file containing all annotations), loads the annotations and saves them in the form of binary masks, respecting [this](#data_struct) structure. It then creates some overview images with each image, its annotations, and the
label description.

>**_NOTE:_** As some scripts demand a number of arguments, there is the option to combine a set of desirable arguments in a text file, and passing it to the script with the @ prefix. However in that case, no wildcards are expanded (as this is done by the shell, and not by the python script)

## Outputs

### Models
Model in [SavedModel](https://www.tensorflow.org/guide/saved_model) or ONNX format.

### Results
- During training, a folder named `intermediate_results` is created, that saves the model output on a random validation image every some epochs.
- A folder `test_images_results_xxx` is created with a figure for each test image, showing the prediction and the groundtruth.
- An entry is added to a `.csv` file named `results_overview.csv` with several performance metrics, overall and per class.
- The confusion matrix of the model is produced.
