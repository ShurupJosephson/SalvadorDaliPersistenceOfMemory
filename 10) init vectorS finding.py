
from sd31 import *
import sys

if __name__ == '__main__':
    try:
        init_vect1 = Vector([GF(2, i) for i in [1, 0, 0, 0, 0]], GF(2, 0), GF(2, 1))
        init_vect2 = Vector([GF(2, i) for i in [1, 0, 0, 0, 0]], GF(2, 0), GF(2, 1))
        bf1 = BF([GF(2, i) for i in [0,0,1,1,1,0,0,1]])
        bf2 = BF([GF(2, i) for i in [1,1,1,0,0,0,1,0]])
        gamma = [GF(2, i) for i in [0,1,0,1,0,0,1,0,1,1,1,1]]

        calc = 0

        while init_vect1 is not None:
            init_vect2 = Vector([GF(2, i) for i in [1, 0, 0, 0, 0]], GF(2, 0), GF(2, 1))
            while init_vect2 is not None:
                calc += 1
                lfsr1 = Lfsr(copy.deepcopy(init_vect1),
                             Polynom([GF(2, i) for i in [1, 0, 1, 0, 0, 1]], GF(2, 0), GF(2, 1)))
                lfsr2 = Lfsr(copy.deepcopy(init_vect2),
                             Polynom([GF(2, i) for i in [1, 0, 0, 1, 0, 1]], GF(2, 0), GF(2, 1)))
                g = []
                for i in range(len(gamma)):
                    g.append(
                        bf1.func([lfsr1.vect[j] for j in [0, 2, 4]]) + bf2.func([lfsr2.vect[j] for j in [0, 2, 4]]))
                    lfsr1.clock()
                    lfsr2.clock()
                if gamma == g:
                    print(f'Initial_1 = ', *init_vect1)
                    print(f'Initial_2 = ', *init_vect2)
                    print(*g)
                    break
                init_vect2 = init_vect2.increment()
            init_vect1 = init_vect1.increment()

        print(calc)


    except ValueError as e:
        print(f'Value error: {e}', file=sys.stderr)
