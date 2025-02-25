import os
import requests
import zipfile

from src.utils.logger import Logger
from src.utils.enums import MovieLensDataset


class Downloader:
    def __init__(self, download_folder="/tmp/dataset", extract_folder="/tmp/dataset"):
        self.download_folder = download_folder
        self.extract_folder = extract_folder
        self.logger = Logger.get_logger(name="Downloader")

        if not os.path.exists(self.download_folder):
            os.makedirs(self.download_folder)
        if not os.path.exists(self.extract_folder):
            os.makedirs(self.extract_folder)

    def download_file(self, url, save_path):
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        self.logger.info(f"Iniciando download de {url}")
        with open(save_path, "wb") as file:
            downloaded = 0
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                downloaded += len(data)
                if total:
                    percent = downloaded / total * 100
                    self.logger.info(f"\rProgresso: {percent:.2f}%", end="")
            self.logger.info("\nDownload concluído.")

    def get_common_folder(self, members):
        folder_names = [member.split('/')[0] for member in members if '/' in member]
        if folder_names and len(set(folder_names)) == 1:
            return folder_names[0] + '/'
        return ''

    def unzip_file(self, zip_path, extract_to=None):
        if extract_to is None:
            extract_to = self.extract_folder
        if not os.path.exists(extract_to):
            os.makedirs(extract_to, exist_ok=True)
        if not zipfile.is_zipfile(zip_path):
            self.logger.info(f"O arquivo {zip_path} não é um arquivo zip válido.")
            return
        self.logger.info(f"Descompactando {zip_path} para {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            common_folder = self.get_common_folder(members)
            if common_folder:
                for member in members:
                    new_path = member[len(common_folder):]  # Remove o prefixo comum
                    if new_path:
                        target_path = os.path.join(extract_to, new_path)
                        if member.endswith('/'):
                            os.makedirs(target_path, exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                                target.write(source.read())
            else:
                zip_ref.extractall(extract_to)
        self.logger.info("Descompactação concluída.")

    def is_extracted(self, dataset: MovieLensDataset) -> bool:
        expected_path = os.path.join(self.extract_folder, dataset.name)
        return os.path.exists(expected_path)

    def download_and_extract_dataset(self, dataset: MovieLensDataset):
        if self.is_extracted(dataset):
            self.logger.info(f"O dataset {dataset.name} já foi extraído. Pulando...")
            return

        url = dataset.value
        filename = os.path.basename(url)
        save_path = os.path.join(self.download_folder, filename)

        self.download_file(url, save_path)

        dataset_extract_path = os.path.join(self.extract_folder, dataset.name)
        self.unzip_file(save_path, extract_to=dataset_extract_path)

    def download_and_extract_all(self):
        for dataset in MovieLensDataset:
            self.download_and_extract_dataset(dataset)


