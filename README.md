[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=ffffff)](https://jupyter.org/)
[![Python3](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3106/)

# Brain Tumour Detection

Application of Machine Learning, AI and Data Mining methods, such as YOLOv8 model and Convolutional Neural Networks (CNNs) for building a model capable of detecting tumours in brain CT scans.

- We used a public dataset available in [Kaggle](https://www.kaggle.com) to develop the project. It's publicly available at the following link: [Medical Image DataSet: Brain Tumor Detection](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection/data);

- We built the project based on a existing Jupyter notebook, also publicly available at Kaggle: [Brain Tumor Detection w/Keras YOLO V8](https://www.kaggle.com/code/banddaniel/brain-tumor-detection-w-keras-yolo-v8); 

- If you want to see the deployed application, click down below and feel free to test the models with your own instances and visualize a static dashboard about the dataset:

     - **Deploy**: [![Deploy](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](https://tumour_detection.streamlit.app/)

# 1. Requirements

- [Python3](https://python.org) and [pip](https://pip.pypa.io/en/stable/installation/) package manager:

      sudo apt install python3 python3-pip build-essential python3-dev
 
- [virtualenv](https://virtualenv.pypa.io/en/latest/) tool:

      pip install virtualenv

- Libraries: [Keras](https://keras.io/), [KerasCV](https://keras.io/keras_cv/), [TensorFlow](https://www.tensorflow.org/?hl=pt-br), [OpenCV](https://opencv.org/), [pandas](https://pandas.pydata.org/), [Streamlit](https://streamlit.io/), [Plotly express](https://plotly.com/python/plotly-express/), [seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/), [numpy](https://numpy.org/), [Pillow](https://pillow.readthedocs.io/en/stable/) and [gdown](https://pypi.org/project/gdown/).

# 2. Web App

In this section, you can see the the detector GUI made with Streamlit.

![Detector](/assets/detector.png)

# 3. Execution

In this section, you can follow detail instructions for executing the project.

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

5. You first need to be in the _src_ directory to run the command:
     
      streamlit run 1_🏠_Home.py
      
# 4. Project Structure

    .
    ├── README.md                             <- Project's documentation
    ├── requirements.txt                      <- File containing all the required dependencies to run the project
    └── src                                   # Directory containing the web application
        ├── 1_🏠_Home.py                      <- Main page with the tumour detector
        └── pages                             # Child pages directory
            └── 2_📊_Static.py                <- Script responsible for generating the static dashboard

# 5. Outro

- To uninstall all dependencies, run the following command:

      pip uninstall -r requirements.txt -y

- To deactivate the virtual environment, run the following command:

      deactivate

- To delete the virtual environment, run the following command:

      rm -rf .venv
