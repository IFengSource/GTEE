"""
Description: Example hyperparameter settings of GTEE across different datasets.

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

from configs.hpo_results.beijing_air import BeijingAir
from configs.hpo_results.ett_h1 import ETT_h1
from configs.hpo_results.ett_m1 import ETT_m1
from configs.hpo_results.wind_turbine import wind_turbine

HPO_RESULTS = {
    "ETT_h1": ETT_h1,
    "BeijingAir": BeijingAir,
    "ETT_m1": ETT_m1,
    "wind_turbine": wind_turbine,
}

__all__ = [
    "HPO_RESULTS",
]
