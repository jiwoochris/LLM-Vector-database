from abc import ABC, abstractmethod

class Interface(ABC) :

    @abstractmethod
    def construct_db(self) :
        pass
    
    
    @abstractmethod
    def generate_text(self) :
        pass