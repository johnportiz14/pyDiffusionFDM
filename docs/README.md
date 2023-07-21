# Building Documentation

## Write test-specific info to page.tex

When creating a new test, add a ``page.tex`` file in that subdirectory. 
For example:

    <REPO>/Cases/<CASE>/page.tex

Can include figures output, verification test comparisons to analytical solutions, etc.

## Run the regression tests

1. Navigate to ``testingPackage/``.
2. Run ``python run_all.py``. 

This calls ``run_case.py`` individually for each test, will then copy relevant ``page.tex`` files and figures over to the ``<REPO>/docs/Cases/<CASE>`` directory.

## Compile documentation

Compile with 

    latexmk -f AllTests.tex

Sometimes you may need to clean up first if no changes seem to be taking placein the compiled PDF:

    latexmk -C AllTests.tex
    latexmk -f AllTests.tex
