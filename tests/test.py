import unittest
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from numfix.numfix import numfix


def test_create_numfix():
    numfix(np.pi)
    numfix([np.pi])
    numfix([np.pi, -np.pi])
    numfix(np.array([np.pi, np.pi]))
    numfix(np.float32(np.pi))
    numfix(666)
    numfix(numfix([1, 2, 3, 4.5]))


def test_swf():
    x = numfix(np.pi, 1, 16, 8)
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 8

    x = numfix([1, 2, 3], 1, 16, 8)[0]
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 8

    x = numfix(np.zeros((3, 3)), 1, 16, 8)[2, 1:3]
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 8

    x = np.arange(10).view(numfix)
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 11

    x = numfix(np.pi)
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 13

    x = numfix()
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 15

    x = numfix([0, 0, 0])
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 15

    x = numfix([0, 1, 0])
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 14

    x = numfix(1024)
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 4

    x = numfix(-1024)
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 5

    x = numfix(102400)
    assert x.s == 1
    assert x.w == 16
    assert x.qf == -2

    x = numfix(0.1024)
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 18

    # self.assertRaises(AssertionError, lambda: numfix(np.pi, 1, 0, 0))  # numfix.w <=0


def test_negtive_f_i():
    x = numfix(0.0547, 0, 8, 10)
    assert x.qf == 10
    assert x.qi  == -2
    assert x.int == 56
    assert x.upper == 0.249023437500000
    assert x.lower == 0

    y = numfix(547, 1, 8, -3)
    assert y.qf == -3
    assert y.qi == 11
    assert y.int == 68
    assert y.upper == 1016
    assert y.lower == -1024


def test_like():
    T = numfix([], 1, 17, 5, RoundingMethod='Floor', OverflowAction='Wrap', FullPrecision=True)
    x = numfix([1, 2, 3, 4], like=T)
    assert x.s == T.s
    assert x.w == T.w
    assert x.qf == T.qf
    assert x.RoundingMethod == T.RoundingMethod
    assert x.OverflowAction == T.OverflowAction
    assert x.FullPrecision == T.FullPrecision
    # test us input_array as like
    y = numfix(x)
    assert y.s == T.s
    assert y.w == T.w
    assert y.qf == T.qf
    assert y.RoundingMethod == T.RoundingMethod
    assert y.OverflowAction == T.OverflowAction
    assert y.FullPrecision == T.FullPrecision


def test_kwargs():
    x = numfix(np.pi, 1, 16, 8, RoundingMethod='Floor', OverflowAction='Wrap', FullPrecision=True)
    assert x.RoundingMethod == 'Floor'
    assert x.OverflowAction == 'Wrap'
    assert x.FullPrecision == True
    # test priority
    y = numfix(np.arange(10), 0, 22, like=x)
    assert y.s == 0
    assert y.w == 22
    assert y.qf == x.qf
    assert y.RoundingMethod == x.RoundingMethod
    assert y.OverflowAction == x.OverflowAction
    assert y.FullPrecision == x.FullPrecision


def test_quantize():
    x = numfix(np.pi, 1, 16, 8)
    assert x == 3.140625000000000
    assert x.bin == '0000001100100100'
    x = numfix(np.pi, 0, 8, 4)
    assert x == 3.125000000000000
    assert x.bin == '00110010'
    x = numfix(1.234567890, 1, 14, 11)
    assert x == 1.234375000000000
    assert x.bin == '00100111100000'
    x = numfix(-3.785, 1, 14, 6)
    assert x == -3.781250000000000
    assert x.bin == '11111100001110'
    x = numfix(0.0123456, 1, 5, 8)
    assert x == 0.011718750000000
    assert x.bin == '00011'


# def test_int64_overflow():
#     self.assertRaises(OverflowError, lambda: numfix([1], 1, 65, 64))


def test_rounding():
    x = numfix(-3.785, 1, 14, 6, RoundingMethod='Nearest')
    assert x == -3.781250000000000
    assert x.bin == '11111100001110'

    x = numfix(-3.785, 1, 14, 6, RoundingMethod='Floor')
    assert x == -3.796875000000000
    assert x.bin == '11111100001101'

    x = numfix(-3.785, 1, 14, 6, RoundingMethod='Ceiling')
    assert x == -3.781250000000000
    assert x.bin == '11111100001110'

    x = numfix(-3.785, 1, 14, 6, RoundingMethod='Zero')
    assert x == -3.781250000000000
    assert x.bin == '11111100001110'


def test_overflow():
    x = numfix([-4, -3, -2, -1, 0, 1, 2, 3], 1, 10, 8, OverflowAction='Saturate')
    assert np.all(x == [-2., -2., -2., -1., 0, 1., 1.996093750000000, 1.996093750000000])
    x = numfix([-4, -3, -2, -1, 0, 1, 2, 3], 0, 10, 8, OverflowAction='Saturate')
    assert np.all(x == [0, 0, 0, 0, 0, 1, 2, 3])
    x = numfix([1, 2, 3], 1, 10, 8, OverflowAction='Wrap')
    assert np.all(x == [1, -2, -1])
    x = numfix([-4 + 4 / 11, -3 + 3 / 11, -2 + 2 / 11, -1 + 1 / 11, 0, 1 + 1 / 11, 2 + 2 / 11, 3 + 3 / 11], 1, 10, 8, OverflowAction='Wrap')
    assert np.all(x == [0.363281250000000, 1.273437500000000, -1.816406250000000, -0.910156250000000, 0, 1.089843750000000, -1.816406250000000, -0.726562500000000])
    x = numfix([1.1, 2.2, 3.3, 4.4, 5.5], 0, 6, 4, OverflowAction='Wrap')
    assert np.all(x == [1.125000000000000, 2.187500000000000, 3.312500000000000, 0.375000000000000, 1.500000000000000])


def test_bin_hex():
    x = numfix(-3.785, 1, 14, 6, RoundingMethod='Zero')
    assert x.bin == '11111100001110'
    assert x.bin_ == '11111100.001110'
    assert x.hex == '-F2'

    y = numfix(547, 1, 8, -3)
    assert y.bin == '01000100'
    assert y.bin_ == '01000100xxx.'

    z = numfix(0.0547, 0, 8, 10)
    assert z.bin == '00111000'
    assert z.bin_ == '.xx00111000'


def test_i():
    x = numfix(-3.785, 1, 14, 6, RoundingMethod='Zero')
    assert x.qi == 8

    x = numfix(3.785, 0, 22, 14, RoundingMethod='Zero')
    assert x.qi == 8


def test_upper_lower_precision():
    x = numfix([], 1, 15, 6)
    assert x.upper == 255.984375
    assert x.lower == -256
    assert x.precision == 0.015625

    x = numfix([], 0, 15, 6)
    assert x.upper == 511.984375
    assert x.lower == 0
    assert x.precision == 0.015625


def test_requantize():
    a = 1.99999999
    s168 = numfix(a, 1, 16, 8)
    s3216 = numfix(a, 1, 32, 16)
    s84 = numfix(a, 1, 8, 4)

    x = numfix(s168)
    y = numfix(x, 1, 32, 16)
    z = numfix(x, 1, 8, 4)
    w = numfix(x, 1, 32, 4)
    u = numfix(x, 1, 32, 28)

    # self.assertAlmostEqual(x, s168)
    # self.assertAlmostEqual(y, x)
    # self.assertAlmostEqual(z, s84)
    # self.assertAlmostEqual(w, s84)
    # self.assertAlmostEqual(u, x)


def test_setitem():
    a = 1.99999999
    s168 = numfix(a, 1, 16, 8)
    s3216 = numfix(a, 1, 32, 16)
    s84 = numfix(a, 1, 8, 4)

    x = numfix(s168)
    x[0] = s3216[0]
    assert x.w == 16
    assert x.qf == 8
    # self.assertAlmostEqual(x, s168)

    x[0] = s84[0]
    assert x.w == 16
    assert x.qf == 8
    # self.assertAlmostEqual(x, s84)


def test_add():
    x = numfix([1, 2, 3, 4], 1, 16, 8)
    x_plus_1 = x + 1
    assert np.all(x_plus_1 == [2, 3, 4, 5])
    assert x_plus_1.w == 17
    assert x_plus_1.qf == 8

    x_plus_y = x + numfix([1.5, 2.5, 3.5, 4.5], 1, 16, 8)
    assert np.all(x_plus_y == [2.5, 4.5, 6.5, 8.5])
    assert x_plus_y.w == 17
    assert x_plus_y.qf == 8

    x += np.int64([256])
    assert np.all(x == [128.99609375, 129.99609375, 130.99609375, 131.99609375])
    assert x.w == 17
    assert x.qf == 8

    z = x + numfix(np.pi, 0, 14, 11)
    assert z.s == 1
    assert z.w == 22
    assert z.qf == 11

    q = x + numfix(np.pi, 1, 14, 11)
    assert q.s == 1
    assert q.w == 21
    assert q.qf == 11


def test_sub():
    x = numfix([1, 2, 3, 4], 1, 16, 8) - 3
    assert np.all(x == [-2, -1, 0, 1])
    assert x.w == 17
    assert x.qf == 8

    y = 3 - numfix([1, 2, 3, 4], 1, 16, 8)
    assert np.all(y == [2, 1, 0, -1])
    assert y.s == 1
    assert y.w == 17
    assert y.qf == 8


def test_not_FullPrecision():
    x = numfix([1, 2, 3, 4], 1, 16, 8, FullPrecision=False)
    y = numfix([2, 3, 4, 5], 1, 17, 9, FullPrecision=False)
    z = numfix(0, 1, 12, 4)

    x1 = x + 1
    assert x1.s == x.s
    assert x1.w == x.w
    assert x1.qf == x.qf

    xy = x + y
    assert xy.s == x.s
    assert xy.w == x.w
    assert xy.qf == x.qf

    y1 = 1 + y
    assert y1.s == y.s
    assert y1.w == y.w
    assert y1.qf == y.qf

    xz = x - z
    assert xz.s == x.s
    assert xz.w == x.w
    assert xz.qf == x.qf

    xy = x * y
    assert xy.s == x.s
    assert xy.w == x.w
    assert xy.qf == x.qf

    y1 = 1 / y
    assert y1.s == y.s
    assert y1.w == y.w
    assert y1.qf == y.qf


def test_mul():
    q = [0.814723686393179, 0.905791937075619, 0.126986816293506]
    a = numfix(q, 1, 16, 8)
    a3 = a * 3
    assert np.all(a3 == [2.449218750000000, 2.718750000000000, 0.386718750000000])
    assert a3.s == 1
    assert a3.w == 32
    assert a3.qf == 21  # note this is different than matlab

    aa = a * numfix(q, 1, 8, 4)
    assert np.all(aa == [0.663330078125000, 0.792968750000000, 0.016113281250000])
    assert aa.w == 24
    assert aa.qf == 12


def test_div():
    q = [0.814723686393179, 0.905791937075619, 0.126986816293506]
    a = numfix(q, 1, 16, 8)
    a3 = a / 0.3333
    assert np.all(a3 == [2.449218750000000, 2.718750000000000, 0.386718750000000])
    assert a3.s == 1
    assert a3.w == 16
    assert a3.qf == 8

    aa = a / numfix(q, 1, 8, 4)
    assert np.all(aa == [1.000000000000000, 1.062500000000000, 1.062500000000000])
    assert aa.w == 16
    assert aa.qf == 4


def test_iop():
    x = numfix(1.12345, 1, 16, 7)
    x += 0.5231
    assert x == 1.648437500000000
    assert x.w == 17
    assert x.qf == 7


def test_fixed_M():
    q = [0.814723686393179, 0.905791937075619, 0.126986816293506]
    a = numfix(q, 1, 16, 8, FullPrecision=True)
    a3 = a / 0.3333
    assert np.all(a3 == [2.449218750000000, 2.718750000000000, 0.386718750000000])
    assert a3.w == 16
    assert a3.qf == 8


def test_neg():
    x = numfix([1, 2, 3], 1, 16, 8)
    assert np.all(-x == [-1, -2, -3])
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 8

    x = numfix([1, 2, 3], 0, 16, 8)
    assert np.all(-x == [0, 0, 0])
    assert x.s == 0
    assert x.w == 16
    assert x.qf == 8


def test_invert():
    x = numfix([1, 2, 3], 1, 16, 8)
    assert np.all(~x == [-1.00390625, -2.00390625, -3.00390625])
    assert x.s == 1
    assert x.w == 16
    assert x.qf == 8

    x = numfix([1, 2, 3], 0, 16, 8)
    assert np.all(~x == [0, 0, 0])
    assert x.s == 0
    assert x.w == 16
    assert x.qf == 8


def test_pow():
    x = numfix([0, 1 + 1 / 77, -3 - 52 / 123], 1, 16, 8)
    y = x**3
    assert np.all(y == [0., 1.03515625, -40.06640625])
    assert y.s == 1
    assert y.w == 16
    assert y.qf == 8


def test_bitwise():
    n = np.array([0b1101, 0b1001, 0b0001, 0b1111]) / 2**8
    x = numfix(n, 1, 16, 8)
    assert np.all((x & 0b1100).int == [0b1100, 0b1000, 0b0000, 0b1100])
    assert np.all((x | 0b0101).int == [0b1101, 0b1101, 0b0101, 0b1111])
    assert np.all((x ^ 0b0110).int == [0b1011, 0b1111, 0b0111, 0b1001])
    assert np.all((x >> 1).int == [0b0110, 0b0100, 0b0000, 0b0111])
    assert np.all((x << 1).int == [0b11010, 0b10010, 0b00010, 0b11110])


def test_logical():
    x = numfix([-2, -1, 0, 1, 2])
    assert np.all((x > 1) == [False, False, False, False, True])
    assert np.all((x >= 1) == [False, False, False, True, True])
    assert np.all((x == 1) == [False, False, False, True, False])
    assert np.all((x != 1) == [True, True, True, False, True])
    assert np.all((x <= 1) == [True, True, True, True, False])
    assert np.all((x < 1) == [True, True, True, False, False])


def test_ufunc():
    x = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]) / 100
    y = numfix(x, 1, 16, 9)
    z = y[0]
    a = np.cos(x)
    b = np.cos(y)
    c = np.cos(y.double)
    d = numfix(c)
    assert np.all(b.int == d.int)
    e = np.arctan2(b, y)
    f = numfix(np.arctan2(b.double, y.double))
    assert np.all(e.int == f.int)
