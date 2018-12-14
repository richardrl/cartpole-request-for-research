import configparser
import os

config = configparser.ConfigParser()
executing_dir = os.path.dirname(os.path.realpath(__file__))
assert os.path.exists(executing_dir + '/config.conf'), executing_dir + '/config.conf'
config.read(executing_dir + '/config.conf')

main_config = config['MAIN']