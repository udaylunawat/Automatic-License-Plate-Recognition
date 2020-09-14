[![Contributors][contributors-shield]][contributors-url] 
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


  <h2 align="center">Indian ALPR System</h>

  <h4 align="center">
  <p align="center">
    Detects License Plates using car images & Deep Learning
    </h4>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Table of Contents](#table-of-contents)
- [Motivation](#motivation)
- [Instructions](#instructions)
- [Getting Started](#getting-started)
  - [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


## Motivation
- The project was primarily made to tackle a myth - "Deep Learning is only useful for Big Data".

## Instructions
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/drive/1BqegosjfXthG1v9p3TUVnfvkvMxAOC5g#scrollTo=LUUvnvqrvFy3"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/udaylunawat/Automatic-License-Plate-Recognition"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://github.com/udaylunawat/Automatic-License-Plate-Recognition"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

<br></br>
## Demo
Link: [Deploy on colab in 2 mins](https://colab.research.google.com/drive/1BqegosjfXthG1v9p3TUVnfvkvMxAOC5g#scrollTo=LUUvnvqrvFy3)

## Home page

<img src="https://media.giphy.com/media/iDxMiptYvfjBqJxdGy/giphy.gif" width="100%" />


YoloV3          |  Retinanet
:-------------------------:|:-------------------------:
![Object detection using Yolo V3](https://media.giphy.com/media/dCRs0OrXbSNmtMRA25/giphy.gif)  |  ![Object detection using Retinanet](https://media.giphy.com/media/UqSyWjDlW5yx9NYg8T/giphy.gif)


Enhance Operations on cropped number plates         |  OCR (Optical Character Recognition)
:-------------------------:|:-------------------------:
![Enhancement on Cropped License Plates](https://media3.giphy.com/media/dZ2ee4W2EGfdEeQTq0/giphy.gif)  |  ![OCR on License Plates](https://media.giphy.com/media/JPVFvEDt8smPbaKdqD/giphy.gif)

<!-- ## Run
> Step 1
### Linux / Mac OS users
### Windows Users
> Step 2

## Deployement on Google Cloud Platform
## Technical Aspect
## Motivation
## Overview
## Installation -->

Directory Tree
------------

    ├── banners                           <- Images for skill banner and project banner
    │
    ├── cfg                               <- Configuration files
    │
    ├── data
    │   ├── sample_images                 <- Sample images for inference
    │   ├── 0_raw                         <- The original, immutable data dump.
    │   ├── 1_external                    <- Data from third party sources.
    │   ├── 2_interim                     <- Intermediate data that has been transformed.
    │   └── 3_processed                   <- The final, canonical data sets for modeling.
    │
    ├── docs                              <- Streamlit / GitHub pages website
    │
    ├── notebooks                         <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                        the creator's initials, and a short `-` delimited description, e.g.
    │                         `              1.0-jqp-initial-data-exploration`.
    │
    ├── output
    │   ├── features                      <- Fitted and serialized features
    │   ├── models                        <- Trained and serialized models, model predictions, or model summaries
    │   │   ├── snapshots                 <- Saving training snapshots.
    │   │   ├── inference                 <- Converted trained model to an inference model.
    │   │   └── TrainingOutput            <- Output logs
    │   └── reports                       <- Generated analyses as HTML, PDF, LaTeX, etc.
    │       └── figures                   <- Generated graphics and figures to be used in reporting
    │
    ├── src                               <- Source code for use in this project.
    │   ├── __init__.py                   <- Makes src a Python module
    │   │
    │   ├── data                          <- Scripts to download or generate data
    │   │   ├── make_dataset.py
    │   │   ├── generate_pascalvoc.py
    │   │   ├── generate_annotations.py
    │   │   └── preprocess.py    
    │   │
    │   ├── features                      <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models                        <- Scripts to train models and then use trained models to make
    │   │   │                                predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization                 <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── utils                             <- Utility scripts for Streamlit, yolo, retinanet etc.
    ├── serve                             <- HTTP API for serving predictions using Streamlit
    │   ├── Dockerfile                    <- Dockerfile for HTTP API
    │   ├── Pipfile                       <- The Pipfile for reproducing the serving environment
    │   └── app.py                        <- The entry point of the HTTP API using Streamlit app
    │
    ├── .dockerignore                     <- Docker ignore
    ├── .gitignore                        <- GitHub's excellent Python .gitignore customized for this project
    ├── app.yaml                          <- contains configuration that is applied to each container started
    │                                        for that service
    ├── config.py                         <- Global configuration variables
    ├── LICENSE                           <- Your project's license.
    ├── Makefile                          <- Makefile with commands like `make data` or `make train`
    ├── README.md                         <- The top-level README for developers using this project.
    ├── tox.ini                           <- tox file with settings for running tox; see tox.readthedocs.io
    ├── requirements.txt                  <- The requirements file for reproducing the analysis environment, e.g.
    │                                        generated with `pip freeze > requirements.txt`
    └── setup.py                          <- makes project pip installable (pip install -e .) so src can be imported


--------
## To Do
1. Convert the app to run without any internet connection.
2. Work with video detection
3. Try AWS Textrac OCR, SSD and R-CNN
4. Try with larger dataset [Google's Open Image Dataset v6](https://storage.googleapis.com/openimages/web/index.html)

## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/udaylunawat/Automatic-License-Plate-Recognition/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/rowhitswami/Indian-Currency-Prediction/issues/new). Please include sample queries and their corresponding results.


## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)  [<img target="_blank" src="https://keras.io/img/logo.png" width=200>](https://keras.io/) [<img target="_blank" src="https://assets.website-files.com/5dc3b47ddc6c0c2a1af74ad0/5e18182ab1025a63c984f248_RGB_Logo_Horizontal_Color_Dark_Bg-p-1600.png" width=200>](https://keras.io/)

[<img target="_blank" src="https://github.com/udaylunawat/Automatic-License-Plate-Recognition/blob/master/banners/docker5.png?raw=true">](https://keras.io/) 

## Team
[![Uday Lunawat](https://avatars1.githubusercontent.com/u/24354945?s=460&u=f1eb1d9248a0287547da38849ffbc0b01c931585&v=4)](https://udaylunawat.github.io/) |
-|
[Uday Lunawat](https://udaylunawat.github.io/) |)

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## License
[![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)](http://www.apache.org/licenses/LICENSE-2.0e)

Copyright 2020 Uday Lunawat

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Credits
- [Rohit Swami]

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

<p align="center"> Made with :blue_heart: for India

[contributors-shield]: https://img.shields.io/github/contributors/udaylunawat/Automatic-License-Plate-Recognition.svg?style=flat-square
[contributors-url]: https://github.com/udaylunawat/Automatic-License-Plate-Recognition/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/udaylunawat/Automatic-License-Plate-Recognition.svg?style=flat-square
[forks-url]: https://github.com/udaylunawat/Automatic-License-Plate-Recognition/network/members

[stars-shield]: https://img.shields.io/github/stars/udaylunawat/Automatic-License-Plate-Recognition.svg?style=flat-square
[stars-url]: https://github.com/udaylunawat/Automatic-License-Plate-Recognition/stargazers

[issues-shield]: https://img.shields.io/github/issues/udaylunawat/Automatic-License-Plate-Recognition.svg?style=flat-square
[issues-url]: https://github.com/udaylunawat/Automatic-License-Plate-Recognition/issues

[license-shield]: https://img.shields.io/github/license/udaylunawat/Automatic-License-Plate-Recognition.svg?style=flat-square
[license-url]: https://github.com/udaylunawat/Automatic-License-Plate-Recognition/blob/master/LICENSE

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/uday-lunawat
