import os
import tarfile
import wget


class FolderStructure:
    def __init__(self) -> None:
        self.raw = "data/raw/"
        self.processed = "data/processed/"
        self.annotations = "data/dataset/annotations/list.txt"
        self.test_annotations = "data/dataset/annotations/test.txt"
        self.train_annotations = "data/dataset/annotations/trainval.txt"

    @staticmethod
    def path_breeds(self):
        self.breeds = f"{self.processed}/breeds/"
        self.train = f"{self.breeds}/train"
        self.test = f"{self.breeds}/test"
        self.train_abyssinian = f"{self.train}/abyssinian"
        self.test_abyssinian = f"{self.test}/abyssinian"
        self.train_bengal = f"{self.train}/bengal"
        self.test_bengal = f"{self.test}/bengal"
        self.train_birman = f"{self.train}/birman"
        self.test_birman = f"{self.test}/birman"
        self.train_bombay = f"{self.train}/bombay"
        self.test_bombay = f"{self.test}/bombay"
        self.train_british_shorthair = f"{self.train}/british_shorthair"
        self.test_british_shorthair = f"{self.test}/british_shorthair"
        self.train_egyptian_mau = f"{self.train}/egyptian_mau"
        self.test_egyptian_mau = f"{self.test}/egyptian_mau"
        self.train_maine_coon = f"{self.train}/maine_coon"
        self.test_maine_coon = f"{self.test}/maine_coon"
        self.train_persian = f"{self.train}/persian"
        self.test_persian = f"{self.test}/persian"
        self.train_ragdoll = f"{self.train}/ragdoll"
        self.test_ragdoll = f"{self.test}/ragdoll"
        self.train_russian_blue = f"{self.train}/russian_blue"
        self.test_russian_blue = f"{self.test}/russian_blue"
        self.train_siamese = f"{self.train}/siamese"
        self.test_siamese = f"{self.test}/siamese"
        self.train_sphynx = f"{self.train}/sphynx"
        self.test_sphynx = f"{self.test}/sphynx"
        self.train_american_bulldog = f"{self.train}/american_bulldog"
        self.test_american_bulldog = f"{self.test}/american_bulldog"
        self.train_american_pitbull = f"{self.train}/american_pit_bull_terrier"
        self.test_american_pitbull = f"{self.test}/american_pit_bull_terrier"
        self.train_basset_hound = f"{self.train}/basset_hound"
        self.test_basset_hound = f"{self.test}/basset_hound"
        self.train_beagle = f"{self.train}/beagle"
        self.test_beagle = f"{self.test}/beagle"
        self.train_boxer = f"{self.train}/boxer"
        self.test_boxer = f"{self.test}/boxer"
        self.train_chihuahua = f"{self.train}/chihuahua"
        self.test_chihuahua = f"{self.test}/chihuahua"
        self.train_english_cocker_spaniel = f"{self.train}/english_cocker_spaniel"
        self.test_english_cocker_spaniel = f"{self.test}/english_cocker_spaniel"
        self.train_english_setter = f"{self.train}/english_setter"
        self.test_english_setter = f"{self.test}/english_setter"
        self.train_german_shorthaired = f"{self.train}/german_shorthaired"
        self.test_german_shorthaired = f"{self.test}/german_shorthaired"
        self.train_great_pyrenees = f"{self.train}/great_pyrenees"
        self.test_great_pyrenees = f"{self.test}/great_pyrenees"
        self.train_havanese = f"{self.train}/havanese"
        self.test_havanese = f"{self.test}/havanese"
        self.train_japanese_chin = f"{self.train}/japanese_chin"
        self.test_japanese_chin = f"{self.test}/japanese_chin"
        self.train_keeshond = f"{self.train}/keeshond"
        self.test_keeshond = f"{self.test}/keeshond"
        self.train_leonberger = f"{self.train}/leonberger"
        self.test_leonberger = f"{self.test}/leonberger"
        self.train_miniature_pinscher = f"{self.train}/miniature_pinscher"
        self.test_miniature_pinscher = f"{self.test}/miniature_pinscher"
        self.train_newfoundland = f"{self.train}/newfoundland"
        self.test_newfoundland = f"{self.test}/newfoundland"
        self.train_pomeranian = f"{self.train}/pomeranian"
        self.test_pomeranian = f"{self.test}/pomeranian"
        self.train_pug = f"{self.train}/pug"
        self.test_pug = f"{self.test}/pug"
        self.train_saint_bernard = f"{self.train}/saint_bernard"
        self.test_saint_bernard = f"{self.test}/saint_bernard"
        self.train_samoyed = f"{self.train}/samoyed"
        self.test_samoyed = f"{self.test}/samoyed"
        self.train_scottish_terrier = f"{self.train}/scottish_terrier"
        self.test_scottish_terrier = f"{self.test}/scottish_terrier"
        self.train_shiba_inu = f"{self.train}/shiba_inu"
        self.test_shiba_inu = f"{self.test}/shiba_inu"
        self.train_staffordshire_bull_terrier = (
            f"{self.train}/staffordshire_bull_terrier"
        )
        self.test_staffordshire_bull_terrier = f"{self.test}/staffordshire_bull_terrier"
        self.train_wheaten_terrier = f"{self.train}/wheaten_terrier"
        self.test_wheaten_terrier = f"{self.test}/wheaten_terrier"
        self.train_yorkshire_terrier = f"{self.train}/yorkshire_terrier"
        self.test_yorkshire_terrier = f"{self.test}/yorkshire_terrier"

    @staticmethod
    def path_cats(self):
        self.cats = f"{self.processed}/cats/"
        self.train = f"{self.cats}/train"
        self.test = f"{self.cats}/test"
        self.train_abyssinian = f"{self.train}/abyssinian"
        self.test_abyssinian = f"{self.test}/abyssinian"
        self.train_bengal = f"{self.train}/bengal"
        self.test_bengal = f"{self.test}/bengal"
        self.train_birman = f"{self.train}/birman"
        self.test_birman = f"{self.test}/birman"
        self.train_bombay = f"{self.train}/bombay"
        self.test_bombay = f"{self.test}/bombay"
        self.train_british_shorthair = f"{self.train}/british_shorthair"
        self.test_british_shorthair = f"{self.test}/british_shorthair"
        self.train_egyptian_mau = f"{self.train}/egyptian_mau"
        self.test_egyptian_mau = f"{self.test}/egyptian_mau"
        self.train_maine_coon = f"{self.train}/maine_coon"
        self.test_maine_coon = f"{self.test}/maine_coon"
        self.train_persian = f"{self.train}/persian"
        self.test_persian = f"{self.test}/persian"
        self.train_ragdoll = f"{self.train}/ragdoll"
        self.test_ragdoll = f"{self.test}/ragdoll"
        self.train_russian_blue = f"{self.train}/russian_blue"
        self.test_russian_blue = f"{self.test}/russian_blue"
        self.train_siamese = f"{self.train}/siamese"
        self.test_siamese = f"{self.test}/siamese"
        self.train_sphynx = f"{self.train}/sphynx"
        self.test_sphynx = f"{self.test}/sphynx"

    @staticmethod
    def path_dogs(self):
        self.dogs = f"{self.processed}/dogs/"
        self.train = f"{self.dogs}/train"
        self.test = f"{self.dogs}/test"
        self.train_american_bulldog = f"{self.train}/american_bulldog"
        self.test_american_bulldog = f"{self.test}/american_bulldog"
        self.train_american_pitbull = f"{self.train}/american_pit_bull_terrier"
        self.test_american_pitbull = f"{self.test}/american_pit_bull_terrier"
        self.train_basset_hound = f"{self.train}/basset_hound"
        self.test_basset_hound = f"{self.test}/basset_hound"
        self.train_beagle = f"{self.train}/beagle"
        self.test_beagle = f"{self.test}/beagle"
        self.train_boxer = f"{self.train}/boxer"
        self.test_boxer = f"{self.test}/boxer"
        self.train_chihuahua = f"{self.train}/chihuahua"
        self.test_chihuahua = f"{self.test}/chihuahua"
        self.train_english_cocker_spaniel = f"{self.train}/english_cocker_spaniel"
        self.test_english_cocker_spaniel = f"{self.test}/english_cocker_spaniel"
        self.train_english_setter = f"{self.train}/english_setter"
        self.test_english_setter = f"{self.test}/english_setter"
        self.train_german_shorthaired = f"{self.train}/german_shorthaired"
        self.test_german_shorthaired = f"{self.test}/german_shorthaired"
        self.train_great_pyrenees = f"{self.train}/great_pyrenees"
        self.test_great_pyrenees = f"{self.test}/great_pyrenees"
        self.train_havanese = f"{self.train}/havanese"
        self.test_havanese = f"{self.test}/havanese"
        self.train_japanese_chin = f"{self.train}/japanese_chin"
        self.test_japanese_chin = f"{self.test}/japanese_chin"
        self.train_keeshond = f"{self.train}/keeshond"
        self.test_keeshond = f"{self.test}/keeshond"
        self.train_leonberger = f"{self.train}/leonberger"
        self.test_leonberger = f"{self.test}/leonberger"
        self.train_miniature_pinscher = f"{self.train}/miniature_pinscher"
        self.test_miniature_pinscher = f"{self.test}/miniature_pinscher"
        self.train_newfoundland = f"{self.train}/newfoundland"
        self.test_newfoundland = f"{self.test}/newfoundland"
        self.train_pomeranian = f"{self.train}/pomeranian"
        self.test_pomeranian = f"{self.test}/pomeranian"
        self.train_pug = f"{self.train}/pug"
        self.test_pug = f"{self.test}/pug"
        self.train_saint_bernard = f"{self.train}/saint_bernard"
        self.test_saint_bernard = f"{self.test}/saint_bernard"
        self.train_samoyed = f"{self.train}/samoyed"
        self.test_samoyed = f"{self.test}/samoyed"
        self.train_scottish_terrier = f"{self.train}/scottish_terrier"
        self.test_scottish_terrier = f"{self.test}/scottish_terrier"
        self.train_shiba_inu = f"{self.train}/shiba_inu"
        self.test_shiba_inu = f"{self.test}/shiba_inu"
        self.train_staffordshire_bull_terrier = (
            f"{self.train}/staffordshire_bull_terrier"
        )
        self.test_staffordshire_bull_terrier = f"{self.test}/staffordshire_bull_terrier"
        self.train_wheaten_terrier = f"{self.train}/wheaten_terrier"
        self.test_wheaten_terrier = f"{self.test}/wheaten_terrier"
        self.train_yorkshire_terrier = f"{self.train}/yorkshire_terrier"
        self.test_yorkshire_terrier = f"{self.test}/yorkshire_terrier"

    @staticmethod
    def path_species(self):
        self.species = f"{self.processed}/species/"
        self.train = f"{self.species}/train"
        self.test = f"{self.species}/test"
        self.train_dog = f"{self.train}/dog"
        self.test_dog = f"{self.test}/dog"
        self.train_cat = f"{self.train}/cat"
        self.test_cat = f"{self.test}/cat"

    def create(self, folder_name=None):
        pass
        # print(self.path_species)
        # path = f"{os.path.abspath(os.getcwd())}/{folder_name}"
        # directory = os.path.dirname(path)
        # try:
        #     os.makedirs(path, exist_ok=True)
        #     print(f"Directory {directory} created successfully.")
        # except OSError:
        #     print(f"Directory {directory} can not be created.")


class Downloader:
    def __init__(self) -> None:
        self.url_imagens = (
            "http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        )
        self.url_annotations = (
            "http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        )

        self.path_to_save = f"{os.path.abspath(os.getcwd())}/{FolderStructure().raw}"

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
    list_attrs = [
        values
        for key, values in FolderStructure().__dict__.items()
        if not key.startswith("__")
    ]
    print(list_attrs)
    # Downloader().images_downloader()
    # Downloader().annotations_downloader()
