# To get modules
from . import app
from . import voice_bot
from . import api
from . import pygpt4all_cli
from . import utils
# To get sub-modules
from .app import *
from .voice_bot import *
from .api import *
from .pygpt4all_cli import *
from .utils import *

#################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number = '0.0.1'
print(f"""{module_type} pyChatGPT_GUI version:{version_number}.""")




