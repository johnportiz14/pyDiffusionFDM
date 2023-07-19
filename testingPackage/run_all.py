#! /usr/bin/env python

#=========================================================================
#||                                                                     ||
#||   This file is part of the                                          ||
#||                                                                     ||
#||   D E M O N S T R A T I O N    Regression Test System               ||
#||   ---------------------------------------------------               ||
#||                                                                     ||
#||   Developed by: Scott R. Runnels, Ph.D.                             ||
#||                                                                     ||
#||            For: CU-Boulder CVEN 5838-200 and -200B                  ||
#||                                                                     ||
#||                 Not for distribution outside of this class          ||
#||                                                                     ||
#||     Copyright 2017-2021 Scott Runnels                               ||
#||                                                                     ||
#||                                                                     ||
#=========================================================================

import os
from os.path import join
import sys
import getopt


def run_all(argv):
    print()
    print("====================================================================")
    print("||                                                                ||")
    print("||  R U N   A L L   T E S T   C A S E S                           ||")
    print("||                                                                ||")
    print("||  Runs all test cases                                           ||")
    print("||                                                                ||")
    print("====================================================================")
    print()

    # ----------------------------------
    # Check environment variables
    # ----------------------------------

    try:
        REPO      = os.environ['REPO']
    except:
        print('Set your REPO env. variable, e.g.:')
        print('    >> export REPO=/project/gas_seepage/jportiz/scripts/pyDiffusionFDM')
        exit(0)

    #  try:
        #  CODE      = os.environ['CODE']
    #  except:
        #  print('Set your CODE env. variable')
        #  exit(0)

    # ----------------------------------------
    # Set location of the run_case.py script
    # ----------------------------------------

    #  RunScript = REPO + '/codes/testingPackage/run_case.py'
    RunScript = REPO + '/testingPackage/run_case.py'
    #  print(RunScript)
    #  RunScript = join(REPO,'/testingPackage/run_case.py')
    #  print(RunScript)

    # ----------------------------------------
    # Create list of tests  
    # ----------------------------------------

    cases = []
    cases.append('diffusionCoupon')
    #  cases.append('test_01')
    #  cases.append('test_02')
    #  cases.append('gradBC_01')
    #  cases.append('multiPhys_01')
    #  cases.append('mms_01')
    #  cases.append('mms_02')

    # ----------------------------------------
    # Loop over tests, executing each one
    # ----------------------------------------

    os.system('rm TestResult_summary')

    f = open(REPO + '/docs/AllTests.tex','w')

    # LaTex Intro Stuff
    print('\\documentclass{article}', file=f)
    print('\\usepackage{graphicx}', file=f)
    print('\\title{pyDiffusionFDM Regression Tests}', file=f)
    print('\\author{John P. Ortiz}', file=f)
    print('\\date{\today}', file=f)
    print('\\begin{document}', file=f)
    print('\\maketitle', file=f)

    for C in cases:
        print('(o) Executing test: ' + C)
        os.system(RunScript + ' -c ' + C )

        # Include each Case as a section
        print('\\section{' + C.replace('_','\_') + '}', file=f)  # Latex syntax test_01 --> test\_01 in latex

        print('\\input{Cases/' + C + '/input.tex}', file=f)

    f.close()






    print()
    print("====================================================================")
    print("||                                                                ||")
    print("||  R E S U L T S                                                 ||")
    print("||                                                                ||")
    print("====================================================================")
    print()
    print()

    os.system('cat TestResult_summary')

    print()
    print()


if __name__=='__main__':

    args = sys.argv[1:]

    run_all(args)


