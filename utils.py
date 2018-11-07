

def abort_clean (error_msg, error_msg2=""):
    '''
    Stops the execution of the program.
    Displays up to 2 messages before exiting.
    '''
    print("ERROR : " + error_msg)
    if error_msg2 :
        print("      : " + error_msg2)
    print(" -- ABORTING EXECUTION --")
    print()
    exit()
