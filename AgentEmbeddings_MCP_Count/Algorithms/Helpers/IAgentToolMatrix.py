from abc import ABC, abstractmethod

class IAgentToolMatrix(ABC):
    # priceHistoty is a dictionary with stock tickers as keys and list of prices as values
    
    @property
    def NumberOfTools(self)->int:
        pass

    @property
    def NumberOfAgents(self)->int:
        pass

    @abstractmethod
    def get_MAT_a_tau(self, pricesHistory:dict)->str:
        pass

  