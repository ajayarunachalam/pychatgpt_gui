#!/bin/sh

chmod ugo+rx ./run_app_mac.sh

sudo easy_install pip

sudo -H pip install virtualenv

# Start Virtual Environment

## create "pychatgpt_ui_test" as a new virtualenv

virtualenv pychatgpt_ui_test

## Activate virtualenv

source pychatgpt_ui_test/bin/activate

## Navigate to where you want to store your code. Create new directory.

git clone https://github.com/ajayarunachalam/pychatgpt_gui.git && cd pychatgpt_gui

# assumes running from root directory
python3 app.py
echo "Launching the pyChatGPT GUI APP.."


