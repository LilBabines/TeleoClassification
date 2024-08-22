import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3",config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(cfg.data.dataset_path)
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print("log_dir: ", log_dir)

if __name__ == "__main__":
    main()