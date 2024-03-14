from abc import ABC
import numpy as np
import torch


class Encoding(ABC):
    """
    A way of encoding data.

    Attributes
    ----------
    encoding : dict
        A dictionary with the symbols to encode as keys and the neural activities as values
    symbols : list
        A list of all possible symbols

    Methods
    -------
    __call__(data):
        Encode a list of symbols
    decode(enc_data):
        Decode a list of neural activities
    """

    def __init__(self, encoding: dict[str, np.ndarray]):
        self.encoding = encoding
        self._update_decoding(encoding)

    def _update_decoding(self, encoding):
        self._decoding = {}
        for key, value in encoding.items():
            self._decoding[self._parse_value(value)] = key

    @property
    def symbols(self) -> list:
        return list(self.encoding.keys())

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        self._encoding = value
        self._update_decoding(value)

    def _parse_symbol(self, symbol):
        if isinstance(symbol, torch.Tensor):
            symbol = float(symbol)
        return symbol

    def _parse_value(self, value):
        if isinstance(value, torch.Tensor):
            value = value.cpu()
        value = np.array(value, dtype=np.float32)
        value = np.round(value, 10)
        value = tuple(value)
        return value

    def __call__(self, data):
        if hasattr(data, "__iter__") and not isinstance(data, str):
            return np.array([self(x) for x in data])
        return self.encoding[self._parse_symbol(data)]

    def decode(self, enc_data):
        if len(enc_data.shape) > 1:
            return np.array([self.decode(x) for x in enc_data])
        return self._decoding[self._parse_value(enc_data)]


class OneHot(Encoding):
    """Encode by sending each symbol to a vector with a single one and zeros everywhere else."""

    def __init__(self, symbols: list):
        encoding = {}
        for i, symbol in enumerate(symbols):
            vector = np.zeros(len(symbols))
            vector[i] = 1
            encoding[symbol] = vector
        super().__init__(encoding)


class Direct(Encoding):
    """Encode an input simply by the given vector, i.e. nothing is changed."""

    def __init__(self):
        self._decoding = lambda x: x
        # self.encoding = lambda x: x

    @property
    def symbols(self) -> list:
        return None

    def _update_decoding(self, encoding):
        pass

    # @property
    # def encoding(self):
    #     return None

    def __call__(self, data):
        return data

    def decode(self, enc_data):
        return enc_data.numpy()
