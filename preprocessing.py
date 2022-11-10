import numpy as np


class Encoding:
    """
    A way of encoding data.

    Attributes
    ----------
    encoding : dict
        A dictionary with the symbols to encode as keys and the neural activities as values

    Methods
    -------
    __call__(data):
        Encode a list of symbols
    decode(enc_data):
        Decode a list of neural activities
    """

    def __init__(self, encoding: dict):
        self.encoding = encoding
        self._update_decoding(encoding)

    def _update_decoding(self, encoding):
        self._decoding = {}
        for key, value in encoding.items():
            self._decoding[tuple(value)] = key

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        self._encoding = value
        self._update_decoding(value)

    def __call__(self, data):
        return [self.encoding[x] for x in data]

    def decode(self, enc_data):
        return [self._decoding[tuple(x)] for x in enc_data]


class OneHot(Encoding):
    """
    Encode by sending each symbol to a vector with a single one and zeros everywhere else.

    Attributes
    ----------
    symbols : list
        An ordered list of symbols
    """

    def __init__(self, symbols: list):
        self.symbols = symbols
        encoding = {}
        for i, symbol in enumerate(symbols):
            vector = np.zeros(len(symbols))
            vector[i] = 1
            encoding[symbol] = vector
        super().__init__(encoding)
