def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   
        elif shell == 'TerminalInteractiveShell':
            return False  
        else:
            return False  
    except NameError:
        return False   