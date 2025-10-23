import dataclasses
from typing import List, Optional
@dataclasses.dataclass
class CAPConfig:
    """
    Configuration for a single evaluation run.
    
    Attributes:
        dataset_name: The name of the dataset to load (e.g "gsm8k", "nq").
        metrics: A list of metrics to compute (e.g ["em", "f1"]).
        dataset_subset: Optional: for datasets with multiple subsets 
                        (e.g "main" for gsm8k, or a specific Longbench task).
        dataset_split: The split to use (e.g "test", "validation").
    """
    dataset_name: str
    metrics: List[str]
    dataset_subset: Optional[str] = None
    dataset_split: str = "test" # Default