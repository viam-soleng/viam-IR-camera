#!/bin/bash
set -e
UNAME=$(uname -s)
  
if [ "$UNAME" = "Linux" ]
then
    echo "Installing venv on Linux"
    sudo apt-get install -y python3-venv
fi
if [ "$UNAME" = "Darwin" ]
then
    echo "Installing venv on Darwin"
    brew install python3-venv
fi

python3 -m venv .venv
. .venv/bin/activate
pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host piwheels.org --trusted-host www.piwheels.org --trusted-host archive1.piwheels.org -r requirements.txt
