import sys
import traceback
from sd31 import *

if __name__ == '__main__':
    try:
        gamma = [GF(2, i) for i in [0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1]]
        init_vect = Vector([GF(2, i) for i in [1, 0, 0, 0, 0, 0]], GF(2, 0), GF(2, 1))
        lfsr = Lfsr(copy.deepcopy(init_vect), Polynom([GF(2, i) for i in [1, 1, 0, 0, 0, 0, 1]], GF(2, 0), GF(2, 1)))

        while init_vect is not None:
            bf_vect = Vector([GF(2, i) for i in [0, 0, 0, 0]], GF(2, 0), GF(2, 1))
            while bf_vect is not None:
                bf = BF(bf_vect.lst)
                g = []
                lfsr.vect = copy.deepcopy(init_vect)
                for i in range(len(gamma)):
                    g.append(bf.func([lfsr.vect[0]] + [lfsr.vect[5]]))
                    lfsr.clock()
                if g == gamma:
                    print(f'Init vect = {init_vect}')
                    print('BF = ', *bf.arr)
                    break
                bf_vect = bf_vect.increment()
            init_vect = init_vect.increment()

    except ValueError as e:
        traceback.print_exc()
        print(f'Value error: {e}', file=sys.stderr)
