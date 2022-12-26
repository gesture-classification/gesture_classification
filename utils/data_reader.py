import numpy as np
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class DataReader():
    """Чтение данных
    Аргумент:
    ----------
    path_to_data (_str_) - путь до данных из конфига
    """    
    def __init__(self, path_to_data):
        super(DataReader, self).__init__()
        self.data = None
        self._load_data_from_path(path_to_data)
        
    
    def _load_data_from_path(self, path_to_data):
        try:
            self.data = np.load(path_to_data, allow_pickle=True)
        
        except Exception as err:
            print('use path to *.npy or *.pkl file')