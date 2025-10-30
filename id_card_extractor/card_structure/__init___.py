# card_structure/__init__.py

# Import all extractor classes
from .adhaar import AadhaarExtractor
from .pan import PANExtractor
from .passport import PassportExtractor

# Define what gets imported with `from card_structure import *`
__all__ = ['AadhaarExtractor', 'PANExtractor', 'PassportExtractor']