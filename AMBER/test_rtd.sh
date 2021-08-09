# Initialize virtualenv
python3.7 -mvirtualenv ./venv/latest/

# Install sphinx-related
./venv/latest/bin/python -m pip install --upgrade --no-cache-dir Pygments==2.3.1 setuptools==41.0.1 docutils==0.14 mock==1.0.1 pillow==5.4.1 commonmark==0.8.1 recommonmark==0.5.0 sphinx==1.8.5 sphinx-rtd-theme==0.4 readthedocs-sphinx-ext==1.0 

# Install package
./venv/latest/bin/python -m pip install --upgrade --upgrade-strategy eager --no-cache-dir .

# Build doc
./venv/latest/bin/sphinx-build -T -d _build/doctrees-readthedocs -D language=en ./docs/source _build/html
