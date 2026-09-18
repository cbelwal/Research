from abc import ABC, abstractmethod

class IUserToolMatrix(ABC):
    # priceHistoty is a dictionary with stock tickers as keys and list of prices as values
    
    @property
    def NumberOfTools(self)->int:
        pass

    @property
    def NumberOfUsers(self)->int:
        pass

    @abstractmethod
    def get_MAT_u_tau(self, pricesHistory:dict)->str:
        pass

  