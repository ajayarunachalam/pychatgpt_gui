#!/bin/sh

# assumes run from root directory
echo "Installing pychatGPT_GUI - A simple, ease-to-use Python GUI Wrapper for unleashing the power of GPT" 
echo "START..."
echo "Installing the dependencies..."
pip install -U -r requirements.txt
echo "End"
echo "Installation Complete"
xterm -e python -i -c "print('>>> from pychatgpt_gui.app import *');from pychatgpt_gui.app import *"
echo "Test Environment Configured"
echo "Package Installed & Tested Sucessfully"

