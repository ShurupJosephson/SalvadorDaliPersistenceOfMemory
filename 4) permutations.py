from sd31 import *


if __name__ == '__main__':
    try:
        p1 = Permutation([1, 5, 2, 4, 3])
        print(f'Ind({p1}) = {p1.ind()}')

        i2 = 31
        print(f'Ind^(-1)({i2}) = {Permutation.permut_from_ind(i2)}')

        p3 = Permutation([1, 3, 2, 5, 4])
        print(f'Vect({p3}) = {p3.invers_vect()}')

        v4 = [0, 1, 1, 1, 2]
        print(f'Vect^(-1)({v4}) = {Permutation.permut_from_inverse_vect(v4)}')

        p5 = Permutation([1, 2, 3, 4, 5])
        print(f'{p5} -> ', end='')
        for i in range(2):
            p5 = p5.lexicographical_next()
            print(f'{p5} -> ', end='')
        print(p5.lexicographical_next())

        p6 = Permutation([4, 3, 1, 5, 2])
        print(f'{p6} = ', end='')
        for i in p6.decompose_cycle():
            print(f'({i}) ', end='')

    except ValueError as e:
        traceback.print_exc()
        print(f'Value error: {e}', file=sys.stderr)
