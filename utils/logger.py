from os.path import join, exists
import numpy as np
from os import mkdir
import time


class Logger:
    """
        Simple logger class
    """
    def __init__(self, data, root='.'):
        """
            Constructor
        :param data: Name of the folder which will contain the time evolution data
        :param root: Root path for the data folder (must exist on the system, is the cwd by default.
        """
        root = Logger.make_new_subdir(root, data)
        self.root = root
        self.timeevo_root = Logger.make_new_subdir(self.root, 'timeevo')
        self.coeff_root = Logger.make_new_subdir(self.root, 'coeff')

    @staticmethod
    def make_new_subdir(root, head):
        """
            Helper method to create a new folder (head) in an existing directory (root)
        """
        full_path = join(root, head)
        if exists(full_path):
            return full_path
        else:
            try:
                mkdir(full_path)
            except FileExistsError:
                time.sleep(0.1)
            return full_path

    @staticmethod
    def save_file(full_path, data):
        """
            Helper method to save data (as int, float, tuple, list or numpy array) using np.savetxt
        """
        if isinstance(data, (int, float)):
            data = np.array([data])
        elif isinstance(data, (tuple, list)):
            data = np.array(data)
        with open(full_path, 'w') as file:
            np.savetxt(file, data)

    @staticmethod
    def _log_arrays(root, **arrays):
        """
            Helper method to save multiple arrays in different .txt files.
            The parameter names correspond to the filenames
        """
        for key, val in arrays.items():
            Logger.save_file(join(root, str(key) + '.txt'), val)

    def log_metadata(self, fname, metadata):
        """
            Logs the contents of a dict (metadata) as string in a file named fname in the data folder.
        """
        with open(join(self.root, fname), 'w') as file:
            file.write(str(metadata))

    def log_timeevo(self, **data):
        """
            Logs the arrays passed as data in the data/timeevo folder.
            The parameter names correspond to the filenames.
        """
        Logger._log_arrays(self.timeevo_root, **data)

    def log_coeff(self, **data):
        """
            Logs the arrays passed as data in the data/coeff folder.
            The parameter names correspond to the filenames.
        """
        Logger._log_arrays(self.coeff_root, **data)
