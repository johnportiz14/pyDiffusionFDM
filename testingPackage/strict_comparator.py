#! /usr/bin/env python

import os
import sys
import getopt
import filecmp


def strict_comparator(argv):

    print()
    print("---------------------------------------------")
    print("|                                           |")
    print("|   S T R I C T   C O M P A R A T O R       |")
    print("|                                           |")
    print("|   Performs a strict, exact match          |")
    print("|   comparison of two files                 |")
    print("|                                           |")
    print("---------------------------------------------")
    print()

    try:
        opts , args = getopt.getopt(argv,"h s: n:")
    except getopt.GetoptError:
        print("Error in user inputs.")
        exit(0)

    new_result = 'NULL'
    std_result = 'NULL'

    for opt, arg in opts:
        if opt == '-h':
            print('-s for standard file -n new output file')
        elif opt == '-n':
            new_result = arg
        elif opt == '-s':
            std_result = arg


    print('(o) Comparing files ' + new_result + ' and ' + std_result)

    # For "Result" to be true, both the std and new files must exist and
    # the must match exactly.

    if os.path.isfile(new_result):
        if os.path.isfile(std_result):
            Result = filecmp.cmp(new_result,std_result)
        else:
            print('std result does not exist')
            Result = False
    else:
        print('new result does not exist')
        Result = False

    # Record the result in a local file called "TestResult_summary"

    f = open("TestResult_summary","w")
    if Result == True:
        print("passed", file=f)
    else:
        print("FAILED", file=f)

    f.close()



if __name__=='__main__':

    args = sys.argv[1:]

    strict_comparator(args)


