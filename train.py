from dataclasses import dataclass
from datetime import datetime
@dataclass
class RunDetails:
    dataset_name: str
    model_id: str
    version: str
    environment: str
    train_state: str
    date: str
