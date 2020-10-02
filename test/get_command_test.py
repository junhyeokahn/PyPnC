import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
from pnc.valkyrie_pnc.valkyrie_interface import ValkyrieInterface

if __name__ == "__main__":

    interface = ValkyrieInterface()
    command = interface.debug_get_command()
