import os
from pathlib import Path

resource_base = Path(__file__).resolve().parent.parent / 'resources'


class Conf:

    @staticmethod
    def paths(name_of_model_or_dataset):
        assert name_of_model_or_dataset in os.listdir(resource_base)
        return resource_base / name_of_model_or_dataset

    @staticmethod
    def show_all_resources():
        resources = os.listdir(resource_base)
        resources = [r for r in resources if os.path.isdir(resource_base / r)]
        print(f"We've got {resources}")
        return resources
