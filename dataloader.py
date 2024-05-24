from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    # Initialize Kaggle API client and authenticate using secrets
    api = KaggleApi()
    api.set_config_value('username', st.secrets["kaggle"]["username"])
    api.set_config_value('key', st.secrets["kaggle"]["key"])
    api.authenticate()
    
    # Define the dataset and the path where files will be downloaded
    dataset = 'borismarjanovic/price-volume-data-for-all-us-stocks-etfs'
    path = 'dataset'

    # Download the dataset
    api.dataset_download_files(dataset, path=path, unzip=True)
