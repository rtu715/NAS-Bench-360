"""
Configurations for TK
"""

from .param_layout import PARAMS_LAYOUT, STATUS_BAR_LAYOUT, NUM_GPUS
import platform

VERSION = "0.2.0"
WELCOME_MSG = r"""
Welcome to BioNAS, an automated machine learning
platform designed for domain experts to build, 
evaluate, and extract knowledge from deep-
learning models.
"""

if platform.system() == 'Darwin':
    LARGE_FONT = ("Helvetica", 15)
    MEDIUM_FONT = ("Helvetica", 12)
    SMALL_FONT = ("Courier", 7)
elif platform.system() == 'Windows':
    LARGE_FONT = ("Times", 15)
    MEDIUM_FONT = ("Times", 12)
    SMALL_FONT = ("Times", 7)
else:
    LARGE_FONT = ("DejaVuSans", 15)
    MEDIUM_FONT = ("DejaVuSans", 12)
    SMALL_FONT = ("DejaVuSans", 7)

MENU_COLOR = 'steelblue'
BODY_COLOR = 'white'
SIDEBAR_COLOR = 'white'
TEXT_BG = 'mint cream'
BTN_BG = 'mint cream'

LOGO = {
    'color': "orange",
    'font': SMALL_FONT,
    'ascii': r"""
$$$$$$$\  $$\           $$\   $$\  $$$$$$\   $$$$$$\
$$  __$$\ \__|          $$$\  $$ |$$  __$$\ $$  __$$\
$$ |  $$ |$$\  $$$$$$\  $$$$\ $$ |$$ /  $$ |$$ /  \__|
$$$$$$$\ |$$ |$$  __$$\ $$ $$\$$ |$$$$$$$$ |\$$$$$$\
$$  __$$\ $$ |$$ /  $$ |$$ \$$$$ |$$  __$$ | \____$$\
$$ |  $$ |$$ |$$ |  $$ |$$ |\$$$ |$$ |  $$ |$$\   $$ |
$$$$$$$  |$$ |\$$$$$$  |$$ | \$$ |$$ |  $$ |\$$$$$$  |
\_______/ \__| \______/ \__|  \__|\__|  \__| \______/


                   Version %s
""" % VERSION
}

REFRESH_INTERVAL = 2500

PARAMS_SHOW_TYPES = (
    'Target Model-Basics',
    'Target Model-Interpret',
    'Controller-Basics',
    'Controller-Environ')

BIONAS_PARAMS = (
    'directory',
    'controller',
    'manager',
    'model_builder',
    'knowledge_func',
    'reward_func',
)

GALLERY_SIZE = (600, 400)
GALLERY_SHOW_TYPES = ('All', 'Overview', 'Operation', 'Inputs', 'Skip-connections')
GALLERY_SHOWTYPE_STARTSWTIH = {
    'All': '',
    'Overview': ('train', 'nas', 'controller'),
    'Operation': 'weight',
    'Inputs': 'inp',
    'Skip-connections': 'skip'}
