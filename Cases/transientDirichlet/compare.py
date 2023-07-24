#! /usr/bin/env python

import os
import sys

if __name__=='__main__':
    REPO = os.environ['REPO']
    #  CODE = os.environ['CODE']

    #----------------------------------------------
    # STANDARD strict_comparator.py TEST
    #----------------------------------------------
    os.system(REPO + '/testingPackage/' + '/strict_comparator.py -s output/concs_.csv_std -n output/concs_.csv')
    #  os.system(REPO + '/codes/' + CODE + '/strict_comparator.py -s phi.plt_std -n phi.plt')

    #----------------------------------------------
    # (OPTIONAL) ADDITIONAL TESTS, e.g., ANALYTICAL VERIFICATION  
    #----------------------------------------------
    print('(o) Running additional plot...')
    os.system('python extra.py')

