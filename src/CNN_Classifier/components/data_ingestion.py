import os
import zipfile
import gdown
from CNN_Classifier import logger
from CNN_Classifier.utils.common import get_size
from CNN_Classifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            dataset_url = self.config.source_URL
            id = dataset_url.split('/')[-1]
            zip_download_dir = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading dataset from {dataset_url} into file: {zip_download_dir}")

            gdown.download(id=id, output=zip_download_dir)
            logger.info(f"Downloaded dataset from {dataset_url} into file: {zip_download_dir}")

        except Exception as e:
            raise e

    def extract_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)