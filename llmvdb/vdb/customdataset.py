import json


class CustomDataset:
    def __init__(self, file_path):
        # Load the JSON data from the file
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        all_data_values = [(i["text"], i["related_to"]) for i in data]

        self.documents_data = all_data_values
