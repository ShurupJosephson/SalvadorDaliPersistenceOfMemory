import sys
import traceback
from sd31 import *


if __name__ == '__main__':
    try:
        gamma1 = [GF(2, i) for i in [0,0,1,0,1,0,0,1,0,1,1,1,1,1,0,1]]
        gamma2 = [GF(2, i) for i in [0,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1]]
        gamma3 = [GF(2, i) for i in [0,0,1,0,1,1,1,1,0,0,1,0,1,1,1,1]]
        init_vect = Vector([GF(2, i) for i in [1, 0, 0, 0, 0, 0, 0, 0, 0]], GF(2, 0), GF(2, 1))
        lfsr = Lfsr(copy.deepcopy(init_vect),
                    Polynom([GF(2, i) for i in [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]], GF(2, 0), GF(2, 1)))
        bf = BF([GF(2, i) for i in [0, 0, 0, 1, 0, 1, 1, 1]])

        flag1 = True
        flag2 = True
        flag3 = True
        while init_vect is not None:
            lfsr.vect = copy.deepcopy(init_vect)
            g = []
            for i in range(len(gamma1)):
                g.append(bf.func([lfsr.vect.lst[0]] + [lfsr.vect.lst[6]] + [lfsr.vect.lst[8]]))
                lfsr.clock()
            if flag1 and g == gamma1:
                print('Init vect 1 = ', *init_vect)
                flag1 = False
            elif flag2 and g == gamma2:
                print('Init vect 2 = ', *init_vect)
                flag2 = False
            elif flag3 and g == gamma3:
                print('Init vect 3 = ', *init_vect)
                flag3 = False
            if not flag1 and not flag2 and not flag3:
                break
            init_vect = init_vect.increment()

        print(
            'AHTUNG! Method just brutforces init vectors to take 3 gammas, but 1 of this gammas isn\'t reachable from Hellman\'s Table')

    except ValueError as e:
        traceback.print_exc()
        print(f'Value error: {e}', file=sys.stderr)
