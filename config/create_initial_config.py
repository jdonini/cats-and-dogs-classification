import os
import tarfile
import wget


class InitialStructure:
    def __init__(self) -> None:
        self.url_imagens = (
            "http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        )
        self.url_annotations = (
            "http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        )
        self.relative_path = "database"
        self.absolute_path = os.path.abspath(os.getcwd())
        self.path = f"{self.absolute_path}/{self.relative_path}"
        self.directory = os.path.dirname(self.path)

    def create_database_folder(self):
        try:
            os.makedirs(self.path, exist_ok=True)
            print(f"Directory {self.directory} created successfully")
        except OSError as error:
            print(f"Directory {self.directory} can not be created")


class Downloader:
    def __init__(self) -> None:
        self.path_to_save = InitialStructure().path
        self.url_imagens = InitialStructure().url_imagens
        self.url_annotations = InitialStructure().url_annotations

    def unizip_tar_file(self):
        tar = tarfile.open(f"{self.path_to_save}/{self.filename}")
        tar.extractall(self.path_to_save)
        tar.close()
        if os.path.exists(f"{self.path_to_save}/{self.filename}"):
            os.remove(f"{self.path_to_save}/{self.filename}")
        else:
            print("The file does not exist")

    def images_downloader(self):
        self.filename = "images.tar.gz"
        if os.path.exists(f"{self.path_to_save}/{self.filename}"):
            print("The file already exists")
        else:
            print("Downloading Images...")
            wget.download(self.url_imagens, self.path_to_save)
        self.unizip_tar_file()

    def annotations_downloader(self):
        self.filename = "annotations.tar.gz"
        if os.path.exists(f"{self.path_to_save}/{self.filename}"):
            print("The file already exists")
        else:
            print("Downloading Annotations...")
            wget.download(self.url_annotations, self.path_to_save)
        self.unizip_tar_file()


if __name__ == "__main__":
    InitialStructure().create_database_folder()
    Downloader().images_downloader()
    Downloader().annotations_downloader()
