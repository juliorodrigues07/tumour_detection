from warnings     import filterwarnings
from keras.models import load_model
from PIL          import Image
from os           import getcwd
import streamlit  as st
import numpy      as np
import cv2        as cv


st.set_page_config(layout="wide", page_title="Tumour Detector", page_icon=":brain:")
filterwarnings('ignore', category=FutureWarning)


@st.cache_data
def cache_model(file_name: str) -> any:

    try:
        model = load_model(f'{getcwd()}/../models/{file_name}.keras')
        return model
    except (IsADirectoryError, NotADirectoryError, FileExistsError, FileNotFoundError):
        print("Model not found or doesn't exists!")
        return None


def validate_input(file: any) -> tuple[bool, np.ndarray] | tuple[bool, str]:
    
    try:
        img = Image.open(file)
        img = np.array(img)
        return True, img
    except AttributeError:
        return False, 'Format not supported!'


def detect_instance(input_data: np.ndarray, model_name: str) -> np.ndarray | str:

    if input_data.shape[2] != 3:
        return 'Image must be RGB.'
    
    # Preprocessing
    img = input_data.copy()
    img = cv.resize(img, (160, 160))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    yolo_model = cache_model(file_name=model_name)
    if yolo_model is None:
        return 'Error loading model!'
    
    detections = yolo_model.predict(img, verbose=0)
    num_detections = detections['num_detections'][0]
    bounding_boxes = detections['boxes'][0]

    # Draw the detections
    for i in range(num_detections):
        xmin, ymin, xmax, ymax = list(map(int, bounding_boxes[i]))
        cv.rectangle(input_data, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    if num_detections == 0:
        input_data = cv.putText(input_data, 'No tumour detected!', (5, 10),
                                cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv.LINE_AA)

    input_data = cv.resize(input_data, (500, 500))
    return input_data


if __name__ == '__main__':

    try:
        st.sidebar.image(f'{getcwd()}/../assets/stock.jpg', width=280)
    except (IsADirectoryError, NotADirectoryError, FileExistsError, FileNotFoundError):
        print("Image not found or doesn't exists!")

    left_col, right_col = st.columns(2)

    # Form header
    form1 = left_col.form(key='options')
    form1.title('Brain Tumour Detector')

    # Uploading file field with status message
    uploaded_file = form1.file_uploader('Upload a brain CT scan', type=['png', 'jpg', 'jpeg'])
    status = form1.container()

    # Model selection box
    slct_model = form1.selectbox(label='Select the desired YOLOv8 model', options=['Base', 'Reduced', 'Balanced'])
    slct_model = slct_model.lower()

    submit_button = form1.form_submit_button('Detect')

    # Some information about the available models
    st.header('Help', divider='rainbow')
    st.markdown('- Base: Trained only with the training dataset (6930 images);')
    st.markdown('- Reduced: Trained only with a reduced version of the training dataset (4000 images);')
    st.markdown('- Balanced: Trained only with a balanced version of the training dataset (14100 images).')

    # Checking inputs and detecting tumours
    check, data = validate_input(file=uploaded_file)
    if submit_button is True and check is True:
        scanned = detect_instance(input_data=data, model_name=slct_model)

        if isinstance(scanned, str):
            status.error(scanned)
        else:
            status.success('Image uploaded successfully!')
            with right_col:
                st.image(scanned)
    
    elif submit_button is True and check is False:
        status.error('First you need to upload an image.')
