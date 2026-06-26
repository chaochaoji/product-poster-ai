from pydantic import BaseModel
from pathlib import Path
import yaml


class DataConfig(BaseModel):
    train_path: str
    test_path: str
    dev_path: str
    class_path: str
    stopwords_path: str

    train_pre_path: str
    test_pre_path: str
    dev_pre_path: str

    random_model_save_path: str
    tf_save_path: str


class AppConfig(BaseModel):
    data: DataConfig


config_path = Path(__file__).parent / 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    raw = yaml.safe_load(f)
config = AppConfig(**raw)

if __name__ == '__main__':
    print(config.data.test_path)