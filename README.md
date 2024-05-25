[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=ffffff)](https://jupyter.org/)
[![Python3](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3106/)

# Tumour Detection

Application of Machine Learning, AI and Data Mining methods, such as YOLOv8 model and Convolutional Neural Networks (CNN) for building a model capable of detecting tumours in CT scans.

# 1. Requirements

- [Python3](https://python.org) and [pip](https://pip.pypa.io/en/stable/installation/) package manager:

      sudo apt install python3 python3-pip build-essential python3-dev
 
- [virtualenv](https://virtualenv.pypa.io/en/latest/) tool:

      pip install virtualenv

- Libraries: [Keras](https://keras.io/), [KerasCV](https://keras.io/keras_cv/), [KerasTuner](https://keras.io/keras_tuner/), [TensorFlow](https://www.tensorflow.org/?hl=pt-br), [imbalanced-learn](https://imbalanced-learn.org/stable/), [OpenCV](https://opencv.org/), [pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [numpy](https://numpy.org/), [gdown](https://pypi.org/project/gdown/) and [google-colab](https://pypi.org/project/google-colab/);

- Environments: [Jupyter](https://jupyter.org/).

# 2. Setting the Environment

1. Clone the repository

       git clone https://github.com/juliorodrigues07/tumour_detection.git

2. Enter the repository's directory

       cd tumour_detection

2. Create a virtual environment

       python3 -m venv .venv

3. Activate the virtual environment

       source .venv/bin/activate

4. Install the dependencies

       pip install -r requirements.txt

# 3. Execution

- To visualize the notebooks online and run them ([Google Colaboratory](https://colab.research.google.com/)), click on the following links:
    -  [EDA](https://colab.research.google.com/drive/1xVpRfmFAg68HilpelpRNWncBJO4wDU6W?usp=sharing);
    -  [Data Mining](https://colab.research.google.com/github/juliorodrigues07/tumour_detection/blob/metrics/brain_tumor_detection_w_keras_yolo_v8.ipynb).
 
- To run the notebooks locally, run the commands in the _notebooks_ directory following the template: `jupyter notebook <file_name>.ipynb`.
  
    - EDA (Exploratory Data Analysis):

          jupyter notebook 1_eda.ipynb

    - Data Mining:

          jupyter notebook brain_tumor_detection_w_keras_yolo_v8.ipynb
      
# 4. Project Structure

    .
    ├── README.md                             <- Project's documentation
    ├── requirements.txt                      <- File containing all the required dependencies to run the project
    ├── docs                                  # Directory containing all the presentation slides about the project      
    |   ├── Transparencies - Partial I.pdf
    |   ├── Transparencies - Partial II.pdf
    |   └── Transparencies - Final.pdf           
    └── notebooks                             # Directory containing project's main jupyter notebook
        ├── 1_eda.ipynb
        └── brain_tumor_detection_w_keras_yolo_v8.ipynb

# 5. Outro

- To uninstall all dependencies, run the following command:

      pip uninstall -r requirements.txt -y

- To deactivate the virtual environment, run the following command:

      deactivate

- To delete the virtual environment, run the following command:

      rm -rf .venv
