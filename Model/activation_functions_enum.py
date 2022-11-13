"""Enum representing implemented non-linearities"""
from enum import Enum


class ValidActivationFunctions(Enum):
    """
    Enum of all implemented activation functions.
    """
    RELU = 'ReLu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'

    @classmethod
    def from_describtion(cls, description: str):
        for func in cls:
            if func.value == description:
                return cls
        raise KeyError(f'Activation function {description} not implemented so far!')

    @classmethod
    def check_valid_func(cls, activation_func: str) -> bool:
        """
        Check if used activation function is valid

        Args:
            - activation_func (str): activation function which should be used

        Returns:
            - (bool): true if function can be used and false otherwise
        """
        valid = False
        for func in cls:
            if func.value == activation_func:
                valid = True
        return valid
