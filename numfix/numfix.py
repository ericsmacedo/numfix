import numpy as np


class numfix_tmp(np.ndarray):
    def __new__(cls,
                array=[],
                s: bool = None,
                w: int = None,
                qf: int = None,
                RoundingMethod: str = None,
                OverflowAction: str = None,
                FullPrecision: bool = True,
                like=None):

        if like is None:
            like = array

        if isinstance(s, (int, float, np.integer, np.floating)):
            s = bool(s)
        else:
            s = getattr(like, 's', True)

        if isinstance(w, (int, float, np.integer, np.floating)):
            w = round(w)
        else:
            w = getattr(like, 'w', 16)

        if isinstance(qf, (int, float, np.integer, np.floating)):
            qf = round(qf)
        else:
            qf = getattr(like, 'qf', cls.get_best_precision(array, s=s, w=w))

        if RoundingMethod is None:
            RoundingMethod = getattr(like, 'RoundingMethod', 'Nearest')

        if OverflowAction is None:
            OverflowAction = getattr(like, 'OverflowAction', 'Saturate')

        if FullPrecision is None:
            FullPrecision = getattr(like, 'FullPrecision', True)

        iarray = cls.__quantize__(array, s=s, w=w, qf=qf,
                                  RoundingMethod=RoundingMethod,
                                  OverflowAction=OverflowAction)
        obj = iarray.view(cls)
        obj._s = s
        obj._w = w
        obj._qf = qf
        obj._RoundingMethod = RoundingMethod
        obj._OverflowAction = OverflowAction
        obj._FullPrecision = FullPrecision

        return obj

    @staticmethod
    def __quantize__(array, s: bool, w: int, qf: int, RoundingMethod: 'str',
                     OverflowAction: str):
        """
        Quantize the input array to a fixed-point representation.

        Parameters
        ----------
        array : array_like or numfix_tmp
            Input array to be quantized. If a numfix_tmp instance is provided,
            it will be converted to a double precision floating-point array.
        s : bool
            Sign bit indication.
        w : int
            Word length of the fixed-point representation.
        qf : int
            Number of fractional bits.
        RoundingMethod : str
            Method used for rounding. Should be a valid rounding method string.
        OverflowAction : str
            Action to be taken in case of overflow. Should be a valid overflow
            action string.

        Returns
        -------
        iarray : ndarray
            Quantized array represented as integers.

        Notes
        -----
        This method performs quantization on the input array based on the specified
        parameters, including rounding and overflow handling.

        """
        qi = w - qf

        is_numfix = isinstance(array, numfix_tmp)

        if is_numfix:
            farray = array.double
        else:
            farray = np.asarray(array, dtype=np.float64)

        if farray.shape == ():
            farray = farray.reshape(1, )

        # Obtain array of integers by shifting all fractional bits to the left
        iarray = farray * 2**qf

        if (is_numfix is False) or qf < array.qf:
            iarray = numfix_tmp.do_rounding(iarray, RoundingMethod)

        if (is_numfix is False) or qi < array.qi:
            iarray = numfix_tmp.do_overflow(iarray, s=s, w=w, qf=qf,
                                           OverflowAction=OverflowAction)

        return iarray

    def __array_finalize__(self, obj):
        """
        Finalize the creation of a new array by copying attributes from another
        array, if available.

        Parameters
        ----------
        obj : ndarray or None
            The array from which attributes are copied. If None, no attributes
            are copied.

        Returns
        -------
        None

        Notes
        -----
        This method is automatically called when creating a new array, allowing
        the copying of attributes from a prototype array, if provided. It is
        particularly useful for maintaining additional information associated
        with the array, such as quantization parameters.

        Attributes
        ----------
        _s : bool
            Sign bit indication. Default is True.
        _qi : int
            Integer part bit-width. Default is 16.
        _qf : int
            Fractional part bit-width. Default is 16.
        _RoundingMethod : str
            Rounding method used during quantization. Default is 'Nearest'.
        _OverflowAction : str
            Overflow action during quantization. Default is 'Saturate'.
        _FullPrecision : bool
            Indicates whether the array is in full precision. Default is True.

        Examples
        --------
        >>> obj = ClassName()
        >>> new_array = obj.copy()
        >>> new_array.__array_finalize__(obj)
        """
        self._s = getattr(obj, 's', True)
        self._w = getattr(obj, 'w', 16)
        self._qf = getattr(obj, 'qf', numfix_tmp.get_best_precision(obj, s=self._s,
                                                                   w=self._w))
        self._RoundingMethod = getattr(obj, 'RoundingMethod', 'Nearest')
        self._OverflowAction = getattr(obj, 'OverflowAction', 'Saturate')
        self._FullPrecision = getattr(obj, 'FullPrecision', True)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = [i.double if isinstance(i, numfix_tmp) else i for i in inputs]
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        return type(self)(results, self.s, self.w) if self.FullPrecision else type(self)(results, like=self)

    def __repr__(self):
        if self.s:
            signed = 's'
        else:
            signed = 'u'
        typename = type(self).__name__
        re = typename + self.double.__repr__()[5:]
        return re + \
            f' {signed}Q{self.qi}.{self.qf}-{self.RoundingMethod[0]}/{self.OverflowAction[0]}'

    def __getitem__(self, key):
        # return class with shape (1,) instead of single int/float value
        value = super().__getitem__(key)
        if isinstance(value, numfix_tmp):
            return value
        else:
            return type(self)(value, like=self)

    def __setitem__(self, key, value):
        quantized = type(self)(value, like=self)
        super().__setitem__(key, quantized)

    def __fixed_arithmetic__(self, func, y):
        raise NotImplementedError("__fixed_arithmetic__")

    @staticmethod
    def do_rounding(iarray, RoundingMethod):
        if RoundingMethod in ('Nearest', 'Round', 'Convergent'):  # round towards nearest integer, this method is faster than np.round()
            iarray[iarray > 0] += 0.5
            iarray[iarray < 0] -= 0.5
        elif RoundingMethod == 'Floor':  # round towards -inf
            iarray = np.floor(iarray)
        elif RoundingMethod == 'Zero':  # round towards zero, fastest rounding method
            pass
        elif RoundingMethod == 'Ceiling':  # round towards +inf
            iarray = np.ceil(iarray)
        else:
            raise ValueError(f"invaild RoundingMethod: {RoundingMethod}")
        return iarray.astype(np.int64)

    @staticmethod
    def do_overflow(iarray, s, w, qf, OverflowAction):
        """
        Handle overflow in a quantized array.

        Parameters
        ----------
        iarray : ndarray
            Array of integers to be checked and adjusted for overflow.
        s : bool
            Sign bit indication.
        w : int
            Word length of the fixed-point representation.
        qf : int
            Number of fractional bits.
        OverflowAction : str
            Action to be taken in case of overflow. Should be 'Error', 'Wrap',
            'Saturate', or a custom action.

        Returns
        -------
        iarray : ndarray
            Array with overflow handled according to the specified action.

        Raises
        ------
        OverflowError
            If OverflowAction is set to 'Error' and overflow occurs.

        ValueError
            If an invalid OverflowAction is provided.

        Notes
        -----
        This method adjusts the input array to handle overflow based on the
        specified action. Overflow can be handled by raising an error,
        wrapping around, saturating at the maximum or minimum representable
        values, or using a custom action.

        Examples
        --------
        >>> iarray = ClassName.do_overflow(iarray, True, 16, 8, 'Saturate')
        """

        iarray = iarray.astype(np.int64)

        print(w)

        # Maximum and minimum values with w bits representation
        if s:
            upper = (1 << (w - 1)) - 1
            lower = -(1 << (w - 1))
        else:
            upper = (1 << w) - 1
            lower = 0

        up = iarray > upper
        low = iarray < lower
        if OverflowAction == 'Error':
            if np.any(up | low):
                raise OverflowError(
                    f"Overflow! array[{iarray.min()} ~ {iarray.max()}] > fi({s},{w},{qf})[{lower} ~ {upper}]")
        elif OverflowAction == 'Wrap':
            mask = (1 << w)
            iarray &= (mask - 1)
            if s:
                iarray[iarray >= (1 << (w - 1))] |= (-mask)
        elif OverflowAction == 'Saturate':
            iarray[up] = upper
            iarray[low] = lower
        else:
            raise ValueError(f"invaild OverflowAction: {OverflowAction}")
        return iarray

    @staticmethod
    def get_best_precision(x, s, w):
        """
        Determine the best precision for quantizing a set of floating-point numbers.

        Parameters
        ----------
        x : array_like
            Input array of floating-point numbers.
        s : int
            Sign bit indication (0 for unsigned, 1 for signed).
        w : int
            Word length of the fixed-point representation.

        Returns
        -------
        best_precision : int
            The optimal number of fractional bits for quantization.

        Notes
        -----
        This static method analyzes the input array of floating-point numbers and
        calculates the optimal number of fractional bits (precision) for quantization.
        It considers the range of values in the array and aims to maximize precision
        while covering the entire range.

        If the array is empty or contains only zeros, the default precision is 15 bits.

        Examples
        --------
        >>> data = [0.5, 1.2, 2.8, -3.5]
        >>> best_precision = ClassName.get_best_precision(data, 1, 16)
        >>> print(best_precision)
        13
        """
        x = np.asarray(x, dtype=np.float64)
        maximum = np.max(x) if np.size(x) else 0
        minimum = np.min(x) if np.size(x) else 0
        if not (maximum == minimum == 0):
            if maximum > -minimum:
                return int(w - np.floor(np.log2(maximum)) - 1 - s)
            else:
                return int(w - np.ceil(np.log2(-minimum)) - s)
        else:
            return 15

    # return ndarray with same shape and dtype = '<Uw' where w=self.w
    def base_repr(self, base=2, frac_point=False):
        """
        Generate a function to represent numbers in a specified base.

        Parameters
        ----------
        base : int, optional
            Base of the representation. Default is 2.
        frac_point : bool, optional
            Flag indicating whether to include a fractional point in the output
            for binary representation. Default is False.

        Returns
        -------
        repr_func : callable
            A function that, when applied to an integer or an array of integers,
            returns a string representation of the numbers in the specified base.

        Notes
        -----
        This function provides a flexible way to represent numbers in different bases.
        For binary representation, the function supports customizable formatting,
        including the option to represent fractional parts with 'x' and include a
        fractional point.
        """
        if base == 2:
            def pretty_bin(i):
                b = np.binary_repr(i, width=self.w)
                if self.qi >= 0:
                    return b[:self.qi].ljust(self.qi, 'x') + '.' + b[self.qi:]
                else:
                    return '.' + b.rjust(self.qf, 'x')
            func = pretty_bin if frac_point else lambda i: np.binary_repr(i, self.w)
        else:
            def func(i):
                return np.base_repr(i, base=base)
        if self.size > 0:
            return np.vectorize(func)(self.int)
        else:
            return np.array([], dtype=f"<U{self.w}")

    s = property(lambda self: self._s)
    qf = property(lambda self: self._qf)
    qi = property(lambda self: self._w - self.qf)
    w = property(lambda self: self._w)
    RoundingMethod = property(lambda self: self._RoundingMethod)
    OverflowAction = property(lambda self: self._OverflowAction)
    FullPrecision = property(lambda self: self._FullPrecision)
    precision = property(lambda self: 2**-self.qf)
    bin = property(lambda self: self.base_repr(2))
    bin_ = property(lambda self: self.base_repr(2, frac_point=True))
    oct = property(lambda self: self.base_repr(8))
    dec = property(lambda self: self.base_repr(10))
    hex = property(lambda self: self.base_repr(16))
    data = property(lambda self: self.double)
    Value = property(lambda self: str(self.double))
    upper = property(lambda self: (2**(self.qi - int(self.s))) - self.precision)
    lower = property(lambda self: -(2**(self.qi - 1)) if self.s else 0)
    ndarray = property(lambda self: self.view(np.ndarray))

    @property
    def int(self):
        raise NotImplementedError("int")

    @property
    def double(self):
        raise NotImplementedError("double")

    def __round__(self, y=0):
        return np.round(self, y)

    def __add__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__add__, y)

    def __radd__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__radd__, y)

    def __iadd__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__iadd__, y)  # force in place use same memory is not worthy

    def __sub__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__sub__, y)

    def __rsub__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__rsub__, y)

    def __isub__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__isub__, y)

    def __mul__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__mul__, y)

    def __rmul__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__rmul__, y)

    def __imul__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__imul__, y)

    def __matmul__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__matmul__, y)

    def __truediv__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__truediv__, y)

    def __rtruediv__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__rtruediv__, y)

    def __itruediv__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__itruediv__, y)

    def __floordiv__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__floordiv__, y)

    def __rfloordiv__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__rfloordiv__, y)

    def __ifloordiv__(self, y):
        return self.__fixed_arithmetic__(np.ndarray.__ifloordiv__, y)

    def __neg__(self):
        return type(self)(-self.double, like=self)

    def __pow__(self, y):
        return type(self)(self.double ** y, like=self)

    def __mod__(self, y):
        return type(self)(self.double % y, like=self)
    # bit wise operation use self.int and convert back

    def __invert__(self):
        return type(self)((~self.int) * self.precision, like=self)  # bitwise invert in two's complement

    def __and__(self, y):
        return type(self)((self.int & y) * self.precision, like=self)

    def __or__(self, y):
        return type(self)((self.int | y) * self.precision, like=self)

    def __xor__(self, y):
        return type(self)((self.int ^ y) * self.precision, like=self)

    def __lshift__(self, y):
        return type(self)((self.int << y) * self.precision, like=self)

    def __rshift__(self, y):
        return type(self)((self.int >> y) * self.precision, like=self)

    def __eq__(self, y):
        return self.double == y

    def __ne__(self, y):
        return self.double != y

    def __ge__(self, y):
        return self.double >= y

    def __gt__(self, y):
        return self.double > y

    def __le__(self, y):
        return self.double <= y

    def __lt__(self, y):
        return self.double < y


class numfix(numfix_tmp):
    """fixed point class that holds float in memory"""
    @staticmethod
    def __quantize__(array, s: bool, w: int, qf: int, RoundingMethod: str,
                     OverflowAction: str):

        array_int = numfix_tmp.__quantize__(array, s=s, w=w, qf=qf,
                                           RoundingMethod=RoundingMethod,
                                           OverflowAction=OverflowAction)
        return array_int * (2**-qf)

    @property
    def int(self):
        return (self.ndarray * 2**self.qf).astype(np.int64)

    @property
    def double(self):
        return self.ndarray

    def __fixed_arithmetic__(self, func, y):

        fi = type(self)
        is_numfix_tmp = isinstance(y, numfix_tmp)

        y_float = y
        s, w, qf, qi = self.s, self.w, self.qf, self.qi

        name = func.__name__[-5:-2]  # last 3 words of operator name
        # __ixxx__ are in place operation, like +=,-=,*=,/=
        in_place = func.__name__[2] == 'i'

        if name == 'add' or name == 'sub':
            if is_numfix_tmp:
                y_fi = y
            else:
                y_fi = fi(y, like=self)
            y_float = y_fi.double

            qf = max(qf, y_fi.qf)
            qi = max(qi, y_fi.qi)

            # If both are signed (or unsigned), QI grow by 1. If not, it can grow by 2
            if s == y_fi.s:
                qi = qi + 1
            else:
                qi = qi + 2
            w = qi + qf
        else:
            if is_numfix_tmp:
                y_fi = y
            else:
                y_fi = fi(y_float, s, w, numfix_tmp.get_best_precision(y_float, s, w))
            y_float = y_fi.double

            if name == 'mul':
                if func.__name__ == '__matmul__':
                    w = self.w + y_fi.w + 1
                else:
                    w = self.w + y_fi.w
                qf = self.qf + y_fi.qf
                qi = w - qf
            elif name == 'div':
                if is_numfix_tmp:
                    w = max(self.w, y_fi.w)
                    qf = self.qf - y_fi.qf
                    qi = w - qf

        float_result = func(self.double, y_float)
        # note that quantization is not needed for full precision mode, new w/f
        # is larger so no precision lost or overflow
        if not (self.FullPrecision or in_place):  # if operator is in-place, bits won't grow
            # if fixed, quantize full precision result to shorter length
            return fi(float_result, like=self)
        elif isinstance(y, numfix_tmp) and not y.FullPrecision:
            return fi(float_result, like=y)
        else:
            return fi(float_result, s, w, qf, like=self)
