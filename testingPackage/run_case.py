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
from glob import glob
import shutil

def run_case(argv):

    print()
    print("--------------------------------------------------------------------")
    print("|                                                                  |")
    print("|   R U N   C A S E                                                |")
    print("|                                                                  |")
    print("|   Runs one test case.                                            |")
    print("|                                                                  |")
    print("--------------------------------------------------------------------")
    print()

    try:
        opts , args = getopt.getopt(argv,"h c:")
    except getopt.GetoptError:
        print("Error in user inputs.")
        exit(0)

    case = 'NULL'

    for opt, arg in opts:
        if opt == '-h':
            print('help...')
        elif opt == '-c':
            case = arg

    print('(o) Running case ' + case)

    # Check environment variables

    try:
        REPO      = os.environ['REPO']
    except:
        print('Set your REPO env. variable')
        exit(0)

    #  try:
        #  CODE      = os.environ['CODE']
    #  except:
        #  print('Set your CODE env. variable')
        #  exit(0)
        #  
        #  CaseDir = REPO + '/codes/' + CODE + '/Cases/' + case   # e.g., fd_forwardEuler/Cases/test_01
    CaseDir = REPO + '/Cases/' + case   # e.g., /Cases/diffusionCoupon

    print('(o) Test case location = ' + CaseDir)

    # -------------------------------------
    # Run the test case
    # -------------------------------------

    #  os.system('mkdir ' + REPO + '/docs/Cases/')                 # Make parent location in the docs directory for all test latex files
    if not os.path.exists( REPO+'/docs/Cases/' ):
        os.mkdir(REPO + '/docs/Cases/')                     # Make parent location in the docs directory for all test latex files

    OriginalLocation = os.getcwd()                              # Record where we are now
    os.chdir(CaseDir)                                           # Move to where the test is
    os.system('./clean')                                        # Clean out old results
    #  os.system(REPO + '/codes/' + CODE + '/fd > tmp')            # Run the code

    #  os.system(REPO + '/Cases/' + case '/diffusion1D.py -i input/inputFile.yml')            # Run the code
    os.system('python '+ REPO + '/Cases/' + case + '/diffusion1D.py -i ' + CaseDir + '/input/inputFile.yml')            # Run the code
    os.system('./compare.py')                                   # Run the comparator

    # Auto documentation
    #  os.system('mkdir ' + REPO + '/docs/Cases/' + case)          # Make a location in the docs directory for this latex file
    if not os.path.exists( REPO+'/docs/Cases/' ):
         os.mkdir(REPO + '/docs/Cases/' + case)          # Make a location in the docs directory for this latex file
    #  os.system('cp page.tex ' + REPO + '/docs/Cases/' + case)   # Copy page.tex to docs dir 
    shutil.copy('page.tex', join(REPO + '/docs/Cases/' + case))   # Copy page.tex to docs dir 
    shutil.copy(join('input','inputFile.yml'), join(REPO + '/docs/Cases/' + case))   # Copy inputFile to docs dir 
    # Copy all figures also to that location
    figList = glob(join('output', '*pdf'))
    for fig in figList:
        os.system('cp '+fig +' '+ REPO + '/docs/Cases/' + case)   # Copy *.pdfs to that location

    #  # Auto plotting
#  
    #  if os.path.isfile('gnuplot_command_file'):
        #  os.system('gnuplot gnuplot_command_file 2> gnuplot_errors')
        #  os.system('cp *.png ' + REPO + '/docs/Cases/' + case)   # Copy input.tex to that location


    os.chdir(OriginalLocation)                                  # Return to original location

    # -------------------------------------
    # Communicate results
    # -------------------------------------
    # Retrieve the results
    g = open(CaseDir + '/TestResult_summary')          # compare.py wrote this file
    Result = g.readlines()[0].replace('\n','')         # read the entire file
    g.close()
    # Report to screen:
    print("Case " + case + ": " + Result)
    # Append result to the global TestResult_summary file
    with open('TestResult_summary','a') as file: file.write('Case ' + case + ': ' + Result + '\n')




    print("--------------------------------------------------------------------")
    print("|                                                                  |")
    print("|   R U N   C A S E                                                |")
    print("|                                                                  |")
    print("|   Done                                                           |")
    print("|                                                                  |")
    print("--------------------------------------------------------------------")






if __name__=='__main__':

    args = sys.argv[1:]

    run_case(args)


