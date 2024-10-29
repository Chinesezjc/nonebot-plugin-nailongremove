from pathlib import Path
from typing import List

from nonebot import get_plugin_config
from pydantic import BaseModel, Field


class Config(BaseModel):
    nailong_model_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "models",
    )
    nailong_list_scenes: List[str] = ["481900070", "852281857"]
    nailong_blacklist: bool = False
    nailong_recall: bool = True
    nailong_tip: str = "再发唐龙把你🐎杀了！"


config = get_plugin_config(Config)
