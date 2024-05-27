import matplotlib.pyplot as plt
import plotly.express    as px
import streamlit         as st
import seaborn           as sns
import pandas            as pd
from os import getcwd


st.set_page_config(layout="wide", page_title="Static Dashboard", page_icon=":bar_chart:")


@st.cache_data
def load_dataset(file_name: str) -> pd.DataFrame | None:

    try:
        df = pd.read_csv(f'{getcwd()}/../datasets/{file_name}.csv')
    except (IsADirectoryError, NotADirectoryError, FileExistsError, FileNotFoundError):
        print("Dataset not found or doesn't exists!")
        return None

    return df


if 'image_statistics' not in st.session_state:
    st.session_state['image_statistics'] = load_dataset(file_name='image_statistics')
if 'labels' not in st.session_state:
    st.session_state['labels'] = load_dataset(file_name='labels')
if 'coords' not in st.session_state:
    st.session_state['coords'] = load_dataset(file_name='coords')

df_status = st.container()
if st.session_state['labels'] is None or st.session_state['image_statistics'] is None or st.session_state['coords'] is None:
    df_status.error('Error loading a dataset!')
    
st.title('Graphs About the Dataset')

if st.session_state['labels'] is not None:

    amount = st.session_state['labels'][['Glioma', 'Meningioma', 'Metastatic', 'No Tumour']].sum()

    fig1 = px.bar(x=['Glioma', 'Meningioma', 'Metastatic', 'No Tumour'], y=amount,
                color=['blue', 'blue', 'blue', 'red'], title='Class Distribution',
                labels={'x': 'Class', 'y': 'Amount'}, width=800, height=600)
    fig1.update_layout(showlegend=False)

    st.plotly_chart(fig1, use_container_width=True)

col1, col2 = st.columns(2)

if st.session_state['image_statistics'] is not None:

    fig1 = plt.figure(figsize=(6, 4))
    pixel_kde = sns.kdeplot(data=st.session_state['image_statistics'], x='Std Dev')
    pixel_kde.set_title('Pixel SD Kernel Density Estimate (KDE)')
    pixel_kde.set_xlabel('Standard Deviation')

    col1.pyplot(fig1, use_container_width=True)

if st.session_state['coords'] is not None:

    fig2 = plt.figure(figsize=(6, 5))
    area_kde = sns.kdeplot(data=st.session_state['coords'], x='Area')
    area_kde.set_title('Tumour Area Kernel Density Estimate (KDE)')
    area_kde.set_xlabel('Area')

    col2.pyplot(fig2, use_container_width=True)
