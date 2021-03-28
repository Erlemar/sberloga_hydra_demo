import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(f'User name: {cfg.user.name}.'
          f'Age: {cfg.user.age}')


if __name__ == "__main__":
    my_app()
