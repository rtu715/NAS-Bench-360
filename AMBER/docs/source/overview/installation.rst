Installation
============

AMBER is developed under Python 3.7 and Tensorflow 1.15.

Please follow the steps below to install AMBER. There are two ways to install `AMBER`: 1) cloning the latest development
from the GitHub repository and install with `Anaconda`; and 2) using `pypi` to install a versioned
release.


Get the latest source code
--------------------------
First, clone the Github Repository; if you have previous versions, make sure you pull the latest commits/changes:

.. code-block:: bash

    git clone https://github.com/zj-zhang/AMBER.git
    cd AMBER
    git pull

If you see `Already up to date` in your terminal, that means the code is at the latest change.

Installing with Anaconda
-------------------------
The easiest way to install AMBER is by ``Anaconda``. It is recommended to create a new conda
environment for AMBER:

.. code-block:: bash

    conda create --file ./templates/conda_amber.linux_env.yml
    python setup.py develop


Installing with Pip
-------------------
As of version `0.1.0`, AMBER is on pypi. In the command-line terminal, type the following commands to get it installed:

.. code-block:: bash

    pip install amber-automl

This will also install the required dependencies automatically. The pip install is still in its beta phase, so if you
encouter any issues, please send me a bug report, and try installing with Anaconda as above.


Testing your installation
-------------------------
You can test if AMBER can be imported to your new `conda` environment like so:

.. code-block:: bash

    conda activate amber
    python -c "import amber"

If no errors pop up, that means you have successfully installed AMBER.

.. todo::

    Run ``unittest`` once its in place.
