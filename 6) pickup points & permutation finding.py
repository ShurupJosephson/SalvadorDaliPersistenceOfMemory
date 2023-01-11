import sys
from sd31 import *

if __name__ == '__main__':
    try:
        init_vect = Vector([GF(2, i) for i in [0,1,0,0,1,1]], GF(2, 0), GF(2, 1))
        lfsr = Lfsr(copy.deepcopy(init_vect), Polynom([GF(2, i) for i in [1, 1, 0, 0, 0, 0, 1]], GF(2, 0), GF(2, 1)))
        bf = BF([GF(2, i) for i in [0,1,0,1,0,1,0,1,1,0,1,0,0,1,1,0]])
        gamma = [GF(2, i) for i in [0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,0,1,1]]
        permut = Permutation([1, 2, 3, 4])

        print()

        for inds in [i for i in sorted(subsets([i for i in range(len(init_vect))]), key=(lambda x: len(x))) if len(i) == (len(init_vect) - len(permut.lst))]:
            p = copy.deepcopy(permut)
            while p is not None:
                g = []
                lfsr.vect = copy.deepcopy(init_vect)
                for j in range(len(gamma)):
                    v = [lfsr.vect[k] for k in range(len(init_vect)) if (k != inds[0] and k != inds[1])]
                    y = copy.deepcopy(v)
                    for k in range(len(p.lst)):
                        y[p.lst[k] - 1] = v[k]
                    g.append(bf.func(y))
                    lfsr.clock()
                if gamma == g:
                    print('Pickup points: ', *[k + 1 for k in range(len(init_vect)) if (k != inds[0] and k != inds[1])])
                    print(f'Permutation:    {p}')
                    break
                p = p.lexicographical_next()

    except ValueError as e:
        print(f'Value error: {e}', file=sys.stderr)
