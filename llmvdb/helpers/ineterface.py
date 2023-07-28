from abc import ABC, abstractmethod


class Interface(ABC):
    @abstractmethod
    def initialize_db(self):
        pass

    @abstractmethod
    def generate_prompt(self):
        pass
    
    @abstractmethod
    def create_completion(self):
        pass
