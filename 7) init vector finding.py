import sys
import traceback
from sd31 import *

if __name__ == '__main__':
    try:
        # init_vect = Vector([GF(2, i) for i in [1,0,1,1,0,1]], GF(2, 0), GF(2, 1))
        # lfsr = Lfsr(copy.deepcopy(init_vect), Polynom([GF(2, i) for i in [1, 1, 0, 0, 0, 0, 1]], GF(2, 0), GF(2, 1)))
        # p = Permutation([1, 2, 3, 4, 5, 6])
        # bf = BF([GF(2, i) for i in [0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1,0,1,1,0,1,0,0,0]])
        # gamma = [GF(2, i) for i in [1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,0]]
        #
        # calc = 0
        #
        # while p is not None:
        #     lfsr.vect = copy.deepcopy(init_vect)
        #     g = []
        #     for i in range(len(gamma)):
        #         v = copy.deepcopy(lfsr.vect.lst)
        #         for k in range(len(p.lst)):
        #             v[p.lst[k] - 1] = lfsr.vect.lst[k]
        #         g.append(bf.func(v))
        #         lfsr.clock()
        #     if gamma == g:
        #         print(f'Permutation: {p} =  {p.ind()}')
        #         calc += 1
        #
        #     p = p.lexicographical_next()
        #
        # print(calc)
        f = open('/Users/shurup/Desktop/coefs.txt', 'w')
        for i in range(256):
            print(1 if random.randint(0, 1) == 0 else -1, file=f)





    except ValueError as e:
        traceback.print_exc()
        print(f'Value error: {e}', file=sys.stderr)
