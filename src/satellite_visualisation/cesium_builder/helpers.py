import yaml
from src import logger

log = logger.get_logger(__name__)


def load_yaml_config(abs_config_file_path):
    try:
        with open(abs_config_file_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        log.info(f"Successfully loaded configuration from: {abs_config_file_path}")
        return config_data
    except FileNotFoundError:
        log.error(f"Configuration file not found: {abs_config_file_path}")
        print(f"ERROR: Configuration file not found: {abs_config_file_path}")
    except yaml.YAMLError as e_yaml:
        log.error(f"Error decoding YAML from configuration file: {abs_config_file_path}\n{e_yaml}")
        print(f"ERROR: Invalid YAML in {abs_config_file_path}: {e_yaml}")
    except Exception as e_conf:
        log.error(f"An unexpected error occurred while loading configuration: {e_conf}")
        print(f"ERROR: Could not load configuration: {e_conf}")
    return None
