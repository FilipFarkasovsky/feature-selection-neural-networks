from multiprocessing.sharedctypes import RawArray
from ctypes import c_char
import numpy as np
from typing import Optional,Tuple

class Dataset:
    def __init__(
        self,
        name: str,
        data: np.ndarray,
        classes: Optional[np.ndarray],
        columns: np.ndarray,
        shared: bool = False
        ):

        self.name = name
        self.shared = shared


        if not shared:
            # Standard storage
            self.data = data
            self.classes = classes
            self.columns = columns
        else:
            # Store metadata
            self._data_shape = data.shape
            self._data_type = data.dtype

            self._classes_shape = classes.shape
            self._classes_type = classes.dtype

            self._columns_shape = columns.shape
            self._columns_type = columns.dtype
            
            # Allocate shared memory
            self.data = RawArray(np.ctypeslib.as_ctypes_type(data.dtype), data.size)
            self.classes = RawArray(c_char, classes.nbytes)
            self.columns = RawArray(c_char, columns.nbytes)

            # Create NumPy views
            np_data = np.frombuffer(self.data, dtype=data.dtype).reshape(data.shape)
            np_classes = np.frombuffer(self.classes, dtype=classes.dtype).reshape(classes.shape)
            np_columns = np.frombuffer(self.columns, dtype=columns.dtype).reshape(columns.shape)

            # Copy data
            np.copyto(np_data, data)
            np.copyto(np_classes, classes)
            np.copyto(np_columns, columns)


    def _get_array(self, buffer, dtype, shape) -> np.ndarray:
        arr = np.frombuffer(buffer, dtype=dtype).reshape(shape)
        arr.flags.writeable = False
        return arr

    def get(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        if not self.shared:
            return self.data, self.classes, self.columns

        data = self._get_array(self.data, self._data_type, self._data_shape)
        classes = self._get_array(self.classes, self._classes_type, self._classes_shape)
        columns = self._get_array(self.columns, self._columns_type, self._columns_shape)

        return data, classes, columns

    def get_instances(self) -> np.ndarray:
        if not self.shared:
            return self.data
        return self._get_array(self.data, self._data_type, self._data_shape)

    def get_classes(self) -> Optional[np.ndarray]:
        if not self.shared:
            return self.classes
        return self._get_array(self.classes, self._classes_type, self._classes_shape)

    def get_column_names(self) -> np.ndarray:
        if not self.shared:
            return self.columns
        return self._get_array(self.columns, self._columns_type, self._columns_shape)

    def get_instances_shape(self) -> Tuple[int, ...]:
        if not self.shared:
            return self.data.shape
        return self._data_shape