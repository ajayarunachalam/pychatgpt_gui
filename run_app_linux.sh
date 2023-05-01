#!/bin/sh

# assumes running from root directory
echo "python version" 
xterm -e python -i -c "print('>>> python -version');python -version"
xterm -e python -i -c "print('>>> python app.py');python app.py"
#python -version
#python app.py
echo "Launching the pyChatGPT GUI APP.."


