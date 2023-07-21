# pyDiffusionFDM

A collection of simple 1D tracer diffusion/heat conduction models written in Python.

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


