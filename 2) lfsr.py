from sd31 import *


if __name__ == '__main__':

    try:
        # to find the Mu(x), divide (div_plnms) Fu(x) by gcd(Fu(x), Фu(x) - generator)) (euclid_polynom)
        # Remember, all operations must be in GF(2), so use GF class
        # (it's universal class, working for all fields, you can try to experiment with polynomial arithmetics
        # in other fields, but your exercise suggests binary field)

        # for example: GF(2), Mu(x) = x^7 + x^5 + x^3 + x^2 + x + 1   =>
        # => plnm = Polynom([GF(2, i) for i in [1, 1, 1, 1, 0, 1, 0]], GF(2, 0), GF(2, 1))

        init_vect = Vector([GF(2, i) for i in [0, 0, 0, 0, 0, 0, 0]], GF(2, 0), GF(2, 1))
        plnm = Polynom([GF(2, i) for i in [1, 0, 1, 1, 1, 0, 1, 1]], GF(2, 0), GF(2, 1))
        lfsr = Lfsr(init_vect, plnm)

        for key, val in lfsr.get_cycle_type().items():
            print(f'{key} : {val}')

    except ValueError as e:
        print(f'Value error: {e}', file=sys.stderr)
