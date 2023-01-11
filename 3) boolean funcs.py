from sd31 import *

if __name__ == '__main__':

    try:
        bf = BF([GF(2, i) for i in [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]])
        print(bf)  # just for example

        # Check BF class methods in sd31.py, it has all you need to complete the task

    except ValueError as e:
        traceback.print_exc()
        print(f'Value error: {e}', file=sys.stderr)
