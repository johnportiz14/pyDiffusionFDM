# pyDiffusionFDM

A collection of simple 1D tracer diffusion/heat conduction models written in Python.

Uses a Tri-diagonal Maxrix Algortithm (a.k.a., a Thomas algorithm) to solve the sparse matrix. Thomas algorithm is used from [ @cbellei
cbellei/TDMAsolver.py
](https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9) under a GNU General Public License. 

Citation
--------

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15428199.svg)](https://doi.org/10.5281/zenodo.15428199)


Full model description can be found in: <>


## Usage

Input files can be created and located in the 

    input/

directory. They follow YAML document formatting. 


## Regression Testing

Test cases are set up in root dir  under the subdirectory:

    Cases/

For the purposes of this document, the root dir will hereafter be referred to as ``REPO``. It is helpful to assign the root dir to an environment variable called ``REPO``.

### How to run regression tests

1. Navigate to ``testingPackage/`` directory. 

2. Run the command:
    
    python run_all.py

3. Result will print to screen, as well as to the file: ``TestResult_summary``.

Individual regression tests can be run using the command:

    python run_case.py -c <Case>

where ``<Case>`` is a valid case name.


## Examples

Example use cases for various problems or articles that use pyDiffusionFDM. 

Individual examples can be run by navigating to a given ``Examples/`` subdirectory and using the command:

    python diffusion1D.py -i input/inputFile.yml

