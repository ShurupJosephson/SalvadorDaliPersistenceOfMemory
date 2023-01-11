import sys
import traceback
from sd31 import *


if __name__ == '__main__':
    try:
        init_vect = Vector([GF(2, i) for i in [0, 0, 0, 0, 0, 0, 0]], GF(2, 0), GF(2, 1))
        lfsr = Lfsr(copy.deepcopy(init_vect), Polynom([GF(2, i) for i in [1, 1, 0, 0, 0, 0, 0, 1]], GF(2, 0), GF(2, 1)))
        bf = BF([GF(2, i) for i in [0,0,1,0,0,1,1,1]])
        gamma = [GF(2, i) for i in [1,1,0,1,1,1,1,1]]

        while init_vect is not None:
            lfsr.vect = copy.deepcopy(init_vect)
            g = []
            for i in range(8):
                g.append(bf.func(lfsr.vect.lst[0:3]))
                lfsr.clock()
            if gamma == g:
                print(f'Init vect: ', init_vect)
                break
            init_vect = init_vect.increment()

    except ValueError as e:
        traceback.print_exc()
        print(f'Value error: {e}', file=sys.stderr)
