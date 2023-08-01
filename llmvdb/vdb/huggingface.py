from datasets import load_dataset

class HuggingFaceDataset:
    
    def __init__(self, huggingface_link : str):
        self.huggingface_link = huggingface_link
        
        # Download Data from huggingface
        data = load_dataset(self.huggingface_link)["train"]
        
        return data