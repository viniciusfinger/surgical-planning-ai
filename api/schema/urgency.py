from enum import Enum

class Urgency(str, Enum):
    elective = "elective"
    urgent = "urgent"
    emergency = "emergency"