# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ImpressionismCat'
copyright = '2022, Ewen Wang'
author = 'Ewen Wang'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon',
              'sphinx.ext.autosectionlabel']

templates_path = ['_templates']
exclude_patterns = []

autoclass_content = 'both'

# Make sure the target is unique
autosectionlabel_prefix_document = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- AutoDoc Mock Imports  ---------------------------------------------------
# autodoc_mock_imports = ["batcat"]

import os
import sys
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))


import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

install('impressionismcat')
import impressionismcat