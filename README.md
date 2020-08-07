Automatic-License-Plate-Recognition-India
==============================

The project has been developed using TensorFlow to detect the License Plates (specifically trained on Indian car dataset) from a car and uses the Tesseract Engine to recognize the charactes from the detected plate.

Software Packages Needed
- Anaconda 3 (Tool comes with most of the required python packages along with python3)
- Tesseract Engine (Must need to be installed)

Python Packages Needed
- Tensorflow
- openCV
- pytesseract


## Method

1. Detect License Plate
2. Perform segmentation of characters
3. Train a ML model to predict characters
4. Prediction of characters in License Plate

## Instructions
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/drive/1A0HUokNvbw37F_1QboYK8UezZcnzsSg4#scrollTo=08X32STHKNS4"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/udaylunawat/Automatic-License-Plate-Recognition"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://github.com/udaylunawat/Automatic-License-Plate-Recognition"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>


<br></br>


Project Organization
------------

    ├── .gitignore         <- GitHub's excellent Python .gitignore customized for this project
    ├── LICENSE            <- Your project's license.
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    ├── data
    │   ├── 0_raw          <- The original, immutable data dump.
    │   ├── 1_external     <- Data from third party sources.
    │   ├── 2_interim      <- Intermediate data that has been transformed.
    │   └── 3_processed    <- The final, canonical data sets for modeling.
    │
    ├── docs               <- Streamlit / GitHub pages website
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── output
    │   ├── features       <- Fitted and serialized features
    │   ├── models         <- Trained and serialized models, model predictions, or model summaries
    │   │   ├── snapshots  <- Saving training snapshots.
    │   │   └── inference  <- Converted trained model to an inference model.
    │   └── reports        <- Generated analyses as HTML, PDF, LaTeX, etc.
    │       └── figures    <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_dataset.py
    │   │   ├── generate_pascalvoc.py
    │   │   ├── generate_annotations.py
    │   │   └── preprocess.py    
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    └── serve                     <- HTTP API for serving predictions using Streamlit
        ├── Dockerfile            <- Dockerfile for HTTP API
        ├── Pipfile               <- The Pipfile for reproducing the serving environment
        └── app.py                <- The entry point of the HTTP API using Streamlit app

--------


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
