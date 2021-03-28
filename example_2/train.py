from dataclasses import dataclass
from typing import Any, Optional

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score


@dataclass
class RFConfig:
    max_depth: Optional[int] = None
    _target_: str = 'sklearn.ensemble.RandomForestClassifier'
    n_estimators: int = 100
    random_state: int = 42


@dataclass
class LogregConfig:
    _target_: str = 'sklearn.linear_model.LogisticRegression'
    penalty: str = 'l1'
    solver: str = 'liblinear'
    C: float = 1.0
    random_state: int = 42
    max_iter: int = 42


@dataclass
class CrossValConfig:
    scoring: Optional[str] = None
    cv: Optional[int] = None


@dataclass
class GeneralConfig:
    random_state: int = 42


@dataclass
class Config:
    # We will populate db using composition.
    model: Any = RFConfig()
    cross_val: CrossValConfig = CrossValConfig()
    general: GeneralConfig = GeneralConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="rf", node=RFConfig)
cs.store(group="model", name="logreg", node=LogregConfig)


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    X, y = load_wine(return_X_y=True, as_frame=True)

    model = instantiate(cfg.model)
    scores = cross_val_score(model, X, y, **cfg.cross_val)
    print(f'Mean score: {np.mean(scores):.4f}. Std: {np.mean(scores):.4f}')


if __name__ == '__main__':
    run()
