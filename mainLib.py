import math
import copy
import collections
import random
from functools import reduce

import numpy as np
from colorama import *





#мхйхрня х ъ опнасел йнлхрхрэ
#мхйхрня х ъ опнасел йнлхрхрэ
#реяр  3

class Matrix:
    def __init__(self, n, m, mtrx=None):
        if n < 1:
            raise ValueError("Invalid rows value")
        if m < 1:
            raise ValueError("Invalid columns value")

        if mtrx is None:
            mtrx = [[0 for i in range(m)] for j in range(n)]
        elif len(mtrx) != n or len(mtrx[0]) != m:
            raise ValueError("Invalid mtrx in init")

        self.n = n
        self.m = m
        self.mtrx = mtrx

    def __str__(self):
        s = ''
        for i in range(self.n):
            for j in range(self.m):
                color = Fore.RESET
                if self.mtrx[i][j] == 1:
                    color = Fore.RED
                s += color + str(self.mtrx[i][j]).rjust(3, ' ')
            s += '\n'
        return s

    def __neg__(self):
        return Matrix(self.n, self.m, [[-i for i in self.mtrx[j]] for j in range(self.m)])

    def __add__(self, other):
        if self.n != other.n or self.m != other.m:
            raise ValueError('Can\'t add mtrxs')

        return Matrix(self.n, self.m, [[a + b for a, b in zip(self.mtrx[i], other.mtrx[i])] for i in range(self.n)])

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if self.m != other.n:
            raise ValueError('Can\'t mult mtrxs')

        res_mtrx = Matrix(self.n, other.m)
        for i in range(res_mtrx.n):
            for j in range(res_mtrx.m):
                res_mtrx.mtrx[i][j] = sum([self.mtrx[i][k] * other.mtrx[k][j] for k in range(self.m)],
                                          self.mtrx[0][0] - self.mtrx[0][0])

        return res_mtrx

    def __eq__(self, other):
        if self.n is not other.n or self.m is not other.m:
            return False
        for i in range(self.n):
            for j in range(self.m):
                if self.mtrx[i][j] is not other.mtrx[i][j]:
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_det(self):
        return np.linalg.det(self.mtrx)

    def set_rand(self, min_val, max_val):
        for i in range(0, self.n):
            for j in range(0, self.m):
                self.mtrx[i][j] = random.randint(min_val, max_val)

    def set_mtrx(self, mtrx_val):
        if len(mtrx_val) != self.n or len(mtrx_val[0]) != self.m:
            raise ValueError("Invalid mtrx in set_mtrx() method")
        self.mtrx = copy.deepcopy(mtrx_val)

    def get_transposed(self):
        return Matrix(self.m, self.n, [[self.mtrx[j][i] for j in range(self.n)] for i in range(self.m)])


def is_prime(n):
    if n == 1:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


class GF:
    def __init__(self, pow, val):
        if pow < 2:
            raise ValueError('Invalid GF power')
        if not is_prime(pow):
            raise ValueError('GF must have prime order')
        if val < 0 or val > pow - 1:
            raise ValueError('Invalid GF elem value')
        self.pow = pow
        self.val = val

    def ord(self):
        i = 1
        val = self.val
        while val != 1:
            val = (val * self.val) % self.pow
            i += 1

        return i

    def is_same_field(self, other):
        return isinstance(other, GF) and self.pow == other.pow

    def __str__(self):
        return str(self.val)

    def __neg__(self):
        return GF(self.pow, (self.pow - self.val) % self.pow)

    def __add__(self, other):
        if not self.is_same_field(other):
            raise ValueError('Invalid fields in elems add')
        return GF(self.pow, (self.val + other.val) % self.pow)

    def __sub__(self, other):
        if not self.is_same_field(other):
            raise ValueError('Invalid fields in elems sub')
        return self + -other

    def __invert__(self):
        r1, r2 = self.val, self.pow
        x1, x2 = 1, 0

        while r2 != 0:
            q = r1 // r2
            r2, r1 = r1 - r2 * q, r2
            x2, x1 = x1 - x2 * q, x2

        return GF(self.pow, (self.pow + x1) % self.pow)

    def __mul__(self, other):
        if not self.is_same_field(other):
            raise ValueError('Invalid fields in elems mul')
        return GF(self.pow, (self.val * other.val) % self.pow)

    def __truediv__(self, other):
        if not self.is_same_field(other):
            raise ValueError('Invalid fields in elems div')
        return self * ~other

    def __pow__(self, power, modulo=None):
        if power == 0:
            return GF(self.pow, 1)
        return reduce(lambda a, b: a * b, [self for i in range(power)])

    def __gt__(self, other):
        if self.is_same_field(other):
            return self.val > other.val
        return self.val > other

    def __lt__(self, other):
        if self.is_same_field(other):
            return self.val < other.val
        return self.val < other

    def __ge__(self, other):
        return self > other or self == other

    def __eq__(self, other):
        if self.is_same_field(other):
            return self.val == other.val
        return self.val == other

    def __ne__(self, other):
        return not self == other

    def __abs__(self):
        return self

    def __int__(self):
        return int(self.val)


class Vector:
    def __init__(self, init_vect, zero_elem, id_elem):
        if len(init_vect) < 1:
            raise ValueError('Invalid vector dimension')
        if type(init_vect[0]) != type(zero_elem) or type(zero_elem) != type(id_elem):
            raise ValueError('Invalid elements types')
        if zero_elem == id_elem:
            raise ValueError('Zero elem can\'t be equal to id elem in field')

        self.lst = init_vect
        self.zero = zero_elem
        self.id = id_elem

    def is_same_field(self, other):
        if type(self) != type(other):
            return False
        if type(self.lst[0]) != type(other.lst[0]):
            return False
        if self.zero != other.zero or self.id != other.id:
            return False
        return True

    def __str__(self):
        return ''.join([str(i) + ' ' for i in self.lst])

    def is_null(self):
        for i in self.lst:
            if i != self.zero:
                return False
        return True

    def increment(self):
        res_vect = copy.deepcopy(self)
        for i in range(len(res_vect.lst)):
            rem, res_vect.lst[i] = res_vect.lst[i], res_vect.lst[i] + res_vect.id
            if rem > res_vect.lst[i]:
                continue
            break

        if res_vect.is_null():
            return None
        return res_vect

    def __getitem__(self, i):
        return self.lst[i]

    def __len__(self):
        return len(self.lst)

    def __neg__(self):
        return Vector([-i for i in self.lst], self.zero, self.id)

    def __add__(self, other):
        if not self.is_same_field(other) or len(self) != len(other):
            raise ValueError('Invalid vector add')
        return Vector([a + b for a, b in zip(self.lst, other.lst)], self.zero, self.id)

    def __sub__(self, other):
        if not self.is_same_field(other) or len(self) != len(other):
            raise ValueError('Invalid vector sub')
        return self + -other

    def __gt__(self, other):
        if not self.is_same_field(other) or len(self) != len(other):
            raise ValueError('Invalid vector compare (>)')
        for i in range(len(self.lst) - 1, 0, -1):
            if self.lst[i] > other.lst[i]:
                return True
            elif self.lst[i] < other.lst[i]:
                return False
        return False

    def __eq__(self, other):
        if not self.is_same_field(other) or len(self) != len(other):
            raise ValueError('Invalid vector compare (==)')
        for i in range(len(self.lst)):
            if self.lst[i] != other.lst[i]:
                return False
        return True

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __lt__(self, other):
        return not self.__ge__(other)

    def __le__(self, other):
        return not self.__gt__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(''.join([str(i) for i in self.lst]))


def div_plnms(p1, p2):
    if p1.zero != p2.zero or p1.id != p2.id:
        raise ValueError('Invalid polynoms divide')
    if p1.deg() < p2.deg():
        return Polynom([p1.zero], p1.zero, p1.id), p1
    if p2.is_null():
        raise ValueError('Null polynom divide')

    p1 = p1.compact()
    p2 = p2.compact()

    res_plnm = Polynom([p1.zero for i in range(len(p1))], p1.zero, p1.id)
    divisible_plnm = p1

    while True:
        # Difference in degree
        deg_diff = divisible_plnm.deg() - p2.deg()
        # Difference in senior coefficient
        coef_diff = divisible_plnm.lst[-1] / p2.lst[-1]

        divisible_plnm = (divisible_plnm - Polynom([coef_diff * i for i in p2.inc_dim(deg_diff).lst], p1.zero,
                                                   p1.id)).compact()
        res_plnm.lst[deg_diff] = coef_diff
        if divisible_plnm.deg() >= p2.deg() and not divisible_plnm.is_null():
            deg1 = divisible_plnm.deg()
            deg2 = p2.deg()
            continue
        break

    return res_plnm, divisible_plnm


def euclid_polynom(p1, p2):
    while (not p1.is_null()) and (not p2.is_null()):
        if p1.deg() > p2.deg():
            p1 = div_plnms(p1, p2)[1]
        else:
            p2 = div_plnms(p2, p1)[1]

    if p1.is_null():
        return p2
    else:
        return p1


class Polynom:
    def __init__(self, init_vect, zero_elem, id_elem):
        if len(init_vect) < 1:
            raise ValueError('Invalid vector dimension')
        if type(init_vect[0]) != type(zero_elem) or type(zero_elem) != type(id_elem):
            raise ValueError('Invalid elements types')
        if zero_elem == id_elem:
            raise ValueError('Zero elem can\'t be equal to id elem in field')

        self.lst = init_vect
        self.zero = zero_elem
        self.id = id_elem

    def is_same_field(self, other):
        if type(self) != type(other):
            return False
        if type(self.lst[0]) != type(other.lst[0]):
            return False
        if self.zero != other.zero or self.id != other.id:
            return False
        return True

    def __str__(self):
        if self.is_null():
            return '0'
        res = ''
        sign_flag = False
        for i in range(len(self) - 1, -1, -1):
            if self.lst[i] == self.zero:
                continue

            if self.lst[i] > 0 and sign_flag:
                res += '+ '
            elif self.lst[i] < 0:
                res += '- '

            sign_flag = True

            if abs(self.lst[i]) != self.id or i == 0:
                if isinstance(self.lst[i], float) and math.modf(self.lst[i])[0] == 0:
                    res += str(int(abs(self.lst[i])))
                else:
                    res += str(abs(self.lst[i]))

            if i > 1:
                res += str(f'x^{str(i)} ')
            elif i == 1:
                res += 'x '

        if res[-1] == ' ':
            res = res[:-1]
        return res

    def __neg__(self):
        return Polynom([-i for i in self.lst], self.zero, self.id)

    def __add__(self, other):
        if not self.is_same_field(other):
            raise ValueError('Invalid polynomials add')

        zero_arr = [self.zero for i in range(abs(len(self.lst) - len(other.lst)))]
        if len(self.lst) > len(other.lst):
            zip_arr = zip(self.lst, other.lst + zero_arr)
        else:
            zip_arr = zip(self.lst + zero_arr, other.lst)

        return Polynom([a + b for a, b in zip_arr], self.zero, self.id)

    def __sub__(self, other):
        if not self.is_same_field(other):
            raise ValueError('Invalid polynomials sub')
        return self + -other

    def __mul__(self, other):
        if isinstance(other, Polynom):
            return sum([(self * i).inc_dim(ind) for ind, i in enumerate(other.lst)],
                       Polynom([self.zero], self.zero, self.id))
        else:
            return Polynom([i * other for i in self.lst], self.zero, self.id)

    def __truediv__(self, other):
        return div_plnms(self, other)

    def __mod__(self, other):
        return div_plnms(self, other)[1]

    def __pow__(self, power, modulo=None):
        if power == 0:
            return Polynom([self.id], self.zero, self.id)
        return reduce(lambda a, b: a * b, [self for i in range(power)])

    def __len__(self):
        return len(self.lst)

    def __key(self):
        return ''.join(str(i) for i in self.compact().lst) + str(self.zero) + str(self.id)

    def __eq__(self, other):
        if isinstance(other, Polynom):
            return self.__key() == other.__key()
        return NotImplemented

    def __hash__(self):
        return hash(self.__key())

    def deg(self):
        for i in range(len(self.lst) - 1, 0, -1):
            if self.lst[i] != self.zero:
                return i
        return 0

    def compact(self):
        return Polynom([self.lst[i] for i in range(self.deg() + 1)], self.zero, self.id)

    def inc_dim(self, k=1):
        return Polynom([self.zero for i in range(k)] + self.lst, self.zero, self.id)

    def is_null(self):
        for i in self.lst:
            if i != self.zero:
                return False
        return True

    def func(self, x):
        res = (x ** 0) * self.lst[0]
        for i in range(1, len(self.lst)):
            res += (x ** i) * self.lst[i]

        return res

    def factor_ring(self):
        if not isinstance(self.id, GF):
            raise ValueError('Factor ring with coefs in infinity field is infinity')

        factor_ring = []
        vect = Vector([self.zero for i in range(self.deg())], self.zero, self.id)

        factor_ring.append(Polynom(vect.lst, self.zero, self.id))
        while vect.increment():
            factor_ring.append(copy.deepcopy(Polynom(vect.lst, self.zero, self.id)))

        return factor_ring

    def is_primitive(self):
        if not isinstance(self.id, GF):
            return NotImplemented

        for plnm in self.factor_ring():
            if (self.func(plnm) % self).is_null():
                plnm_degs = set()
                for i in range(1, self.id.pow ** self.deg()):
                    plnm_degs.add(plnm ** i % self)
                if len(plnm_degs) == self.id.pow ** self.deg() - 1:
                    return True

        return False


class Lfsr:
    def __init__(self, init_vect, init_plnm):
        if init_plnm.deg() != len(init_vect):
            raise ValueError("Invalid state & polynomial")
        if init_vect.zero != init_plnm.zero or init_vect.id != init_plnm.id:
            raise ValueError('Vector and polynom have different fields')

        self.plnm = init_plnm
        self.vect = init_vect

    def clock(self):
        res_bit = self.vect.lst[0]
        new_bit = sum([a * b for a, b in zip(self.vect.lst, self.plnm.lst[:-1])], self.vect.zero)
        self.vect.lst = [self.vect.lst[ind] for ind in range(1, len(self.vect))] + [new_bit]
        return res_bit

    def get_cycle_type(self):
        lfsr = copy.deepcopy(self)
        vect_dim = copy.deepcopy(self.vect)
        vect_set = set()
        res_dict = collections.defaultdict(int)

        while vect_dim:
            vect_dim = vect_dim.increment()
            if vect_dim not in vect_set:
                lfsr.vect = copy.deepcopy(vect_dim)
                vect_set.add(lfsr.vect)
                print(f'{len(vect_set)}    {lfsr.vect}')
                lfsr.clock()
                i = 1
                while lfsr.vect != vect_dim:
                    vect_set.add(copy.deepcopy(lfsr.vect))
                    print(f'{len(vect_set)}    {lfsr.vect}')
                    i += 1
                    lfsr.clock()

                print()
                res_dict[i] += 1

        res_dict[1] += 1  # null vect
        return res_dict


def subsets(arr):
    subs = [[]]
    for i in arr:
        subs += [s + [i] for s in subs]
    return subs


class BF:
    def __init__(self, vect):
        if len(vect) < 1:
            raise ValueError('Vector size < 1')
        if not math.log(len(vect), 2).is_integer():
            raise ValueError('Invalid vector. Size must be = 2^(args number)')
        for i in vect:
            if isinstance(i, GF):
                if i.pow != 2:
                    raise ValueError('Not GF(2) elem in vector')
            else:
                raise ValueError('Not GF(2) elem in vector')

        self.arr = vect
        self.args_num = int(math.log(len(vect), 2))

    def __str__(self):
        return ''.join([format(i, f'>0{self.args_num}b') + ' ' + str(self.arr[i]) + '\n' for i in range(len(self.arr))])

    def func(self, args):
        if len(args) != self.args_num:
            raise ValueError('Invalid args number')
        for i in args:
            if isinstance(i, GF):
                if i.pow != 2:
                    raise ValueError('Not GF(2) elem in vector')
            else:
                raise ValueError('Not GF(2) elem in vector')

        return self.arr[int(''.join([str(i.val) for i in args]), 2)]

    def zhegalcin_plnm(self):
        res_arr = [self.arr[0]]

        buff_arr = copy.deepcopy(self.arr)
        for i in range(len(self.arr) - 1):
            buff_arr = [buff_arr[i] + buff_arr[i + 1] for i in range(len(buff_arr) - 1)]
            res_arr.append(buff_arr[0])

        return res_arr

    def deg(self):
        res = 0
        zheg_plnm = self.zhegalcin_plnm()
        for i in range(len(zheg_plnm)):
            if zheg_plnm[i] == GF(2, 1):
                res = max(res, bin(i).count('1'))
        return res

    def weight(self):
        return self.arr.count(GF(2, 1))

    def fix_func(self, vals, inds):
        if len(vals) != len(inds):
            raise ValueError('Different size of values arr and indexes arr')
        for i in vals:
            if isinstance(i, GF):
                if i.pow != 2:
                    raise ValueError('Not GF(2) elem in values arr')
            else:
                raise ValueError('Not GF(2) elem in values arr')

        for i in inds:
            if i < 0 or i >= self.args_num:
                raise ValueError('Invalid index in indexes arr')

        func_vals = []
        for i in range(len(self.arr)):
            arg = format(i, f'>0{self.args_num}b')
            if math.prod([int(arg[inds[j]] == str(vals[j])) for j in range(len(inds))]) == 1:
                func_vals.append(self.arr[i])

        return BF(func_vals)

    def analytic_struct(self):
        a = [i for i in range(self.args_num)]

        for inds in sorted([i for i in subsets(a) if 0 < len(i) < self.args_num], key=lambda x: (len(x), x[0])):
            print(f'deg f{[i + 1 for i in inds]}: ')
            for val in range(len(self.arr)):
                print(self.fix_func([GF(2, int(format(val, f'>0{self.args_num}b')[i])) for i in inds], inds).deg(),
                      end='  ')
            print()

    def weight_struct(self):
        a = [i for i in range(self.args_num)]

        for inds in sorted([i for i in subsets(a) if 0 < len(i) < self.args_num], key=lambda x: (len(x), x[0])):
            print(f'weight f{[i + 1 for i in inds]}: ')
            for val in range(len(self.arr)):
                print(self.fix_func([GF(2, int(format(val, f'>0{self.args_num}b')[i])) for i in inds], inds).weight(),
                      end='  ')
            print()

    def fourier_trans(self, arr):
        if len(arr) == 1:
            return arr

        l_arr = arr[:int(len(arr) / 2)]
        r_arr = arr[int(len(arr) / 2):]

        l_arr, r_arr = [a + b for a, b in zip(l_arr, r_arr)], [a - b for a, b in zip(l_arr, r_arr)]

        return self.fourier_trans(l_arr) + self.fourier_trans(r_arr)

    def walsh_hadamard_trans(self):
        return self.fourier_trans([int(math.pow(-1, int(i))) for i in self.arr])


def factorial_coefs(num):
    res = []
    q = num

    k = 1
    while math.factorial(k) <= num:
        k += 1
    k -= 1

    for i in range(k, 0, -1):
        fact = math.factorial(i)
        res.insert(0, q // fact)
        q %= fact

    return res


class Permutation:
    def __init__(self, vect):
        if len(set(vect)) != len(vect) or sum(1 for i in vect if i < 1 or i > len(vect)) != 0:
            raise ValueError('Invalid vect in permutation init')
        self.lst = vect

    def __invert__(self):
        p = copy.deepcopy(self.lst)
        for i in range(len(self.lst)):
            p[self.lst[i] - 1] = i + 1
        return Permutation(p)

    def __str__(self):
        return (''.join([str(i) + ' ' for i in self.lst]))[:-1]

    def __mul__(self, other):
        return Permutation([other.func(i) for i in self.lst])

    @staticmethod
    def permut_from_inverse_vect(vect):
        inds = [i + 1 for i in range(len(vect))]
        p = []
        for i in range(len(vect)):
            p.insert(0, inds[-vect[-i - 1] - 1])
            inds.pop(-vect[-i - 1] - 1)

        return Permutation(p)

    @staticmethod
    def permut_from_ind(ind):
        if ind < 0:
            raise ValueError('Invalid number in setting from index')

        p = [1]
        for i in factorial_coefs(ind):
            p = [1 + i] + [j + 1 if j >= 1 + i else j for j in p]

        return Permutation(p)

    @staticmethod
    def cycle_permut(n, k, d):
        if k < 1 or n < 1 or k > n:
            raise ValueError('Invalid args in cycle permutation')
        d %= k
        id = [i + 1 for i in range(n)]
        return Permutation(id[:k][d:] + id[:k][:d] + id[k:])

    def func(self, arg):
        if arg > len(self.lst) or arg < 1:
            raise ValueError('Invalid argument in permutation')
        return self.lst[arg - 1]

    def ind(self):
        if len(self.lst) == 1:
            return 0
        return (self.lst[0] - 1) * math.factorial(len(self.lst) - 1) + Permutation(
            [i - 1 if i > self.lst[0] else i for i in self.lst[1:]]).ind()

    def invers_vect(self):
        return [sum(1 for i in self.lst[:self.lst.index(x)] if i > x) for x in self.lst]

    def lexicographical_next(self):
        p = copy.deepcopy(self.lst)
        k = -1
        for i in range(len(p) - 2, -1, -1):
            if p[i] < p[i + 1]:
                k = i
                break
        if k == -1:
            return None
        j = p.index(min([i for i in p[k + 1:] if p[k] < i]))
        p[k], p[j] = p[j], p[k]
        return Permutation(p[:k + 1] + p[k + 1:][::-1])

    def decompose_cycle(self):
        n = len(self.lst)
        p_list = []

        for i in range(n, 1, -1):
            q = (~self).lst[i - 1]
            for k in range(n, i, -1):
                q = p_list[n - k].lst[q - 1]
            p_list.append(Permutation.cycle_permut(n, i, i - q))

        return p_list
