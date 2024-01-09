from datasets import load_dataset

class CustomDataset:
    def __init__(self, file_path):

        if file_path.endswith(".json") or file_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=file_path)
        else:
            data = load_dataset(file_path)

        all_data_values = [i["text"] for i in data]

        self.documents_data = all_data_values
