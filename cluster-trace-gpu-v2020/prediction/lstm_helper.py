from typing import Dict, List
import pandas as pd
import numpy as np


class Machine():
    
    def __init__(self, machine_name: str) -> None:
        self.machine_name: str = machine_name