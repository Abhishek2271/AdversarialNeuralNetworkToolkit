import os

class DatasetManager:
    def __init__(self):
        self.default_location = "./DataSets/data/"
        self.dataset_directory = None

    def set_dataset_directory(self, path):
        if os.path.isdir(path):
            self.dataset_directory = path
        else:
            raise NotADirectoryError(f"The path {path} is not a valid directory.")

    def get_dataset_directory(self):
        if self.dataset_directory is not None:
            return self.dataset_directory
        else:
            return self.default_location
            #raise ValueError("Dataset directory has not been set.")

# Usage
dataset_manager = DatasetManager()

# Set the dataset directory
dataset_manager.set_dataset_directory('/path/to/your/dataset')

# Get the dataset directory
try:
    directory = dataset_manager.get_dataset_directory()
    print(f"The dataset directory is set to: {directory}")
except ValueError as e:
    print(e)