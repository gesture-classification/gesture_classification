import numpy as np
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class DataPathException(Exception):
    pass


class DataReader:
    """Чтение данных
    Аргумент:
    ----------
    path_to_data (_str_) - путь до данных из конфига
    """    
    def __init__(self, path_to_data: str):
        super().__init__()
        self.path_to_data = path_to_data
        self.data = self._load_data_from_path()
        
    def _load_data_from_path(self) -> np.ndarray:
        try:
            data = np.load(self.path_to_data, allow_pickle=True)
            return data
        
        except DataPathException:
            print('use path to *.npy or *.pkl file')
