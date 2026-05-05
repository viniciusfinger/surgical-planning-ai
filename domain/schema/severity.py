from enum import Enum


class Severity(str, Enum):
    mild = "mild"
    moderate = "moderate"
    severe = "severe"