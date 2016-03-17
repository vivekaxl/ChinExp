#!/bin/bash

VENV_DIR=.venv
PYTHON_VERSION=3.4

action_prepare ()
{
    if [ ! -d "$VENV_DIR" ]; then
        virtualenv $VENV_DIR --python=python$PYTHON_VERSION
    fi
}

action_package_install ()
{
    pip install --upgrade pip
    pip install numpy sympy matplotlib scipy pandas
    pip install jupyter
    pip install pyzmq jinja2 pygments bokeh
    pip install cython https://github.com/scikit-learn/scikit-learn/archive/master.zip
    pip install brewer2mpl prettyplotlib
}

action_dependency_install ()
{
    # from http://unix.stackexchange.com/questions/6345/how-can-i-get-distribution-name-and-version-number-in-a-simple-shell-script
    if [ -f /etc/debian_version ]; then
        sudo apt-get install build-essential python3-dev g++
        sudo apt-get install -y libblas-dev liblapack-dev gfortran
        sudo apt-get install -y libfreetype6-dev libpng-devexit
    elif [ -f /etc/redhat-release ]; then
        sudo yum groupinstall -y 'development tools'
        sudo yum install -y zlib-dev openssl-devel sqlite-devel bzip2-devel
        sudo yum install lapack-devel blas-devel
    else
        echo "This is something else"
    fi
}

action_init ()
{
    #action_dependency_install
    action_prepare
    . $VENV_DIR/bin/activate
    action_package_install
    deactive

}

### Main

action_init
