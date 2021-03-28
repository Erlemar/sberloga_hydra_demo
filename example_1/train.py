import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def run(cfg: DictConfig) -> None:
    """
    Calculate scores.

    Args:
        cfg: hydra config
    """

    X, y = load_wine(return_X_y=True, as_frame=True)
    if cfg.model.name == 'rf':

        rf = RandomForestClassifier(**cfg.model.params)
        print(rf.get_params())
        scores = cross_val_score(rf, X, y)
        print(f'Mean score: {np.mean(scores):.4f}. Std: {np.mean(scores):.4f}')

    elif cfg.model.name == 'logreg':

        lr = LogisticRegression(**cfg.model.params)
        scores = cross_val_score(lr, X, y)
        print(f'Mean score: {np.mean(scores):.4f}. Std: {np.mean(scores):.4f}')


@hydra.main(config_path='conf', config_name='config')
def run_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == '__main__':
    run_model()
