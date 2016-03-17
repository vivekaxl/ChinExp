#!/bin/ash

VENV_DIR=.venv
PYTHON_VERSION=2.6

action_prepare ()
{
    if [ ! -d "$VENV_DIR" ]; then
        virtualenv $VENV_DIR --python=python$PYTHON_VERSION
    fi
    . .venv/bin/activate
    #pip install executor 2>&1 > /dev/null
    deactivate
}

action_init ()
{
    action_prepare
    #bash --rcfile .venv/bin/activate
    . $VENV_DIR/bin/activate
}

### Main

action_init
