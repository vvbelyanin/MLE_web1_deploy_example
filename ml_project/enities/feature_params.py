from dataclasses import dataclass
from typing import List, Optional

@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    binary_features: List[str]
    useless_features: List[str]
    target: Optional[str]
