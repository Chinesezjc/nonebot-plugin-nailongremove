import nonebot
from pathlib import Path
from nonebot.adapters.onebot.v11 import Adapter

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(Adapter)

nonebot.load_plugin(Path("./nonebot_plugin_nailongremove"))
nonebot.run(host="127.0.0.1", port=8080)
