import pdb

def runTest( test, name=None ):
    if name is None:
        name = test.__name__

    try:
        test()
    except:
        print( "FAIL:", name )
        pdb.post_mortem()
        return False
    print( "PASS:", name )
    return True

width = 85

def runSuite( test ):
    '''test should be list of (test, name) tuples'''
    nPass = 0
    nFail = 0
    print( "-" * width )
    for t in test:
        if runTest( *t ):
            nPass += 1
        else:
            nFail += 1
    print( "-" * width )
    print( "Pass:", nPass )
    print( "Fail:", nFail )
    print( "-" * width )