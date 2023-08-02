from datasets import load_dataset


class HuggingFaceDataset:
    def __init__(self, huggingface_link: str):
        self.huggingface_link = huggingface_link

        # Download Data from huggingface
        dataset = load_dataset(self.huggingface_link)["train"]

        documents_data = self.data_features(dataset)

        return documents_data

    def data_features(self, dataset):
        if set(dataset.features.keys()) == set(["documents"]):
            return dataset

        if set(dataset.features.keys()) == set(["instruction", "input", "output"]):
            # Add a 'documents' column to the dataset
            new_dataset = dataset.map(
                lambda dataset: {
                    "documents": f'{dataset["instruction"]}\n\n{dataset["output"]}'
                },
                remove_columns=["instruction", "input", "output"],
            )

            return new_dataset
