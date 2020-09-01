"""
This module can create affine forms and perform operations.

Affine Arithmetic (AA) has been developed to overcome the error explosion
problem of standard Interval Arithmetic (IA).
This method represents a quantity :math:`x` as an affine form :math:`\\hat{x}`,
which is a first degree polynomial:

.. math ::
    \\hat{x} = x_0 + \\sum_{i=1}^{n} x_i\\epsilon_i

The coefficients :math:`x_i` are finite floating-point numbers: they are called
the *partial deviations*.

The coefficient :math:`x_0` is the *central value* of the affine form
:math:`\\hat{x}`.

The :math:`\\epsilon_i` coefficients are symbolic real values called
*noise symbols*. Their values are unknown between -1 and 1.

This representation enables a better tracking of the different quantities
inside the affine form.

For example, the quantity :math:`[0, 10]` can be represented as the following
affine form:

.. math ::
    A = [0, 10] = 5 + 5\\epsilon_1

where

.. math ::
    x_0 = 1, x_1 = 5

But we could also represent it like this:

.. math ::
    B = [0, 10] = 5 + 3\\epsilon_1 + 2\\epsilon_2

where

.. math ::
    x_0 = 5, x_1 = 3, x_2 = 2

Both forms represent the same quantity but they are handling differently the
storage of internal quantities. They will behave differently during operation:

.. math::
    A - A = 0

whereas

.. math::
    A - B = 0 + 2\\epsilon_1 - 2\\epsilon_2 = [0, 4]

The second example illustrates this behaviour. Even though :math:`A` and
:math:`B` represent the same quantity, they manage their quantity differently.
Therefore, they are not equal.

"""
import affapy.ia
import sympy
from affapy.error import affapyError
import mpmath
from mpmath import (
    mp, fdiv, fadd, fsub, fsum, fneg, fmul, fabs, sqrt, exp, log, sin)


class Affine:
    """
    Representation of an affine form.
    An instance of the class **Affine** is composed of three fields:

    * **interval**: the interval associated to the affine form
    * **x0**: the center
    * **xi**: the dictionnary of noise symbols

    """
    _weightCount = 1
    _approxMethod = 'minRange' #Alternatively, minCheb

    def __init__(self, interval=None, x0=None, xi=None):
        """
        Create an affine form. There are two different ways:

        .. code-block:: python

            x1 = Affine(interval=[inf, sup])
            x2 = Affine(x0=0, xi={})

        If no arguments, x0=0 and xi={}.

        The first method is easier to use. To convert an interval
        :math:`[a, b]` into an affine form, there is the formula:

        .. math ::
            \\hat{x} = x_0 + x_k\\epsilon_k

        with:

        .. math ::
            x_0 = \\frac{a + b}{2} ,
            x_k = \\frac{a - b}{2}

        To convert an affine form into an interval :math:`X`:

        .. math ::
            X = [x_0 + rad(x), x_0 - rad(x)]

        with:

        .. math ::
            rad(x) = \\sum_{i=1}^{n} |x_i|

        Args:
            interval (list or tuple with length 2 or Interval): the interval
            x0 (int or float or mpf): the center
            xi (dict of mpf values): noise symbols

        Returns:
            Affine: affine form

        Raises:
            affapyError: interval must be list, tuple or Interval

        Examples:
            >>> from affapy.aa import Affine
            >>> Affine([1, 3])
            Affine(2.0, {5: mpf('-1.0')})
            mpf('1.0')
            >>> print(Affine(x0=1, xi={1:2, 2:3}))
            1.0 + 2.0e1 + 3.0e2

        """
        if interval is not None:
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                inf, sup = min(interval), max(interval)
            elif isinstance(interval, affapy.ia.Interval):
                inf, sup = interval.inf, interval.sup
            else:
                raise affapyError("interval must be list, tuple or Interval")
            self._x0 = (inf + sup) / 2
            self._xi = {Affine._getNewXi(): fdiv(
                fsub(inf, sup, rounding='u'), 2, rounding='u')}
            self._interval = affapy.ia.Interval(inf, sup)
        elif x0 is not None and xi is not None:
            self._x0 = mp.mpf(x0)
            self._xi = {i: mp.mpf(xi[i], rounding='u') for i in xi}
            self._interval = affapy.ia.Interval(
                fadd(self._x0, self.rad()),
                fsub(self._x0, self.rad()))
        else:
            self._x0 = mp.mpf(x0)
            self._xi = {}
            self._interval = affapy.ia.Interval(0, 0)

    # Getter
    @property
    def x0(self):
        """Return the center x0."""
        return self._x0

    @property
    def xi(self):
        """Return the dictionnary of noice symbols xi."""
        return self._xi.copy()

    @property
    def interval(self):
        """Return interval associated to the affine form."""
        return self._interval.copy()

    # Setter
    @x0.setter
    def x0(self, val):
        """
        Set the center x0.
        It updates the interval associated to the affine form.
        """
        self._x0 = mp.mpf(val)
        self._interval = self.convert()

    @xi.setter
    def xi(self, val):
        """
        Set the dictionnary of noice symbols xi.
        It updates the interval associated to the affine form.
        """
        self._xi = {i: mp.mpf(val[i], rounding='u') for i in val}
        self._interval = self.convert()

    @staticmethod
    def _getNewXi():
        """Get a new noise symbol."""
        Affine._weightCount += 1
        return Affine._weightCount - 1

    def rad(self):
        """
        Return the radius of an affine form:

        .. math ::
            rad(x) = \\sum_{i=1}^{n} |x_i|

        Args:
            self (Affine): operand

        Returns:
            mpf: sum of abs(xi)

        Examples:
            >>> x = Affine([1, 3])
            >>> x.rad()
            mpf('1.0')

        """
        return fsum(self.xi.values(), absolute=True)

    # Unary operator
    def __neg__(self):
        """
        **Operator - (unary)**

        Return the additive inverse of an affine form:

        .. math ::
            -\\hat{x} = -x_0 + \\sum_{i=1}^{n} -x_i\\epsilon_i

        Args:
            self (Affine): operand

        Returns:
            Affine: -self

        Examples:
            >>> print(-Affine([1, 2]))
            -1.5 + 0.5e1

        """
        x0 = -self.x0
        xi = {i: fneg(self.xi[i], rounding='u') for i in self.xi}
        return Affine(x0=x0, xi=xi)

    # Affine operations
    def __add__(self, other):
        """
        **Operator +**

        Add two affine forms:

        .. math ::
            \\hat{x} + \\hat{y} =
            (x_0 + y_0) + \\sum_{i=1}^{n} (x_i + y_i)\\epsilon_i

        Or add an affine form and an integer or float or mpf:

        .. math ::
            \\hat{x} + y =
            (x_0 + y) + \\sum_{i=1}^{n} x_i\\epsilon_i

        Args:
            self (Affine): first operand
            other (Affine or int or float or mpf): second operand

        Returns:
            Affine: self + other

        Raises:
            affapyError: other must be Affine, int, float, mpf

        Examples:
            >>> print(Affine([0, 1]) + Affine([3, 4]))
            4.0 + -0.5e1 + -0.5e2
            >>> print(Affine([1, 2]) + 3)
            4.5 + -0.5e1

        """
        if isinstance(other, self.__class__):
            x0 = self.x0 + other.x0
            xi = {}
            for i in self.xi:
                if i in other.xi:
                    val = fadd(self.xi[i], other.xi[i], rounding='u')
                    if val != 0:
                        xi[i] = val
                else:
                    xi[i] = self.xi[i]
            for i in other.xi:
                if i not in self.xi:
                    xi[i] = other.xi[i]
            return Affine(x0=x0, xi=xi)
        if isinstance(other, (int, float, mpmath.mpf)):
            x0 = self.x0 + mp.mpf(other)
            xi = self.xi.copy()
            return Affine(x0=x0, xi=xi)
        raise affapyError("other must be Affine, int, float, mpf")

    def __radd__(self, other):
        """
        **Reverse operator +**

        Add two affine forms or an affine form and an integer or float or mpf.
        See the add operator for more details.

        Args:
            self (Affine): second operand
            other (Affine or int or float or mpf): first operand

        Returns:
            Affine: other + self

        Raises:
            affapyError: other must be Affine, int, float, mpf

        Examples:
            >>> print(1 + Affine([1, 2]))
            2.5 + -0.5e3

        """
        return self + other

    def __sub__(self, other):
        """
        **Operator -**

        Subtract two affine forms:

        .. math ::
            \\hat{x} - \\hat{y} =
            (x_0 - y_0) + \\sum_{i=1}^{n} (x_i - y_i)\\epsilon_i

        Or subtract an affine form and an integer or float or mpf:

        .. math ::
            \\hat{x} - y =
            (x_0 - y) + \\sum_{i=1}^{n} x_i\\epsilon_i

        Args:
            self (Affine): first operand
            other (Affine or int or float or mpf): second operand

        Returns:
            Affine: self - other

        Raises:
            affapyError: other must be Affine, int, float, mpf

        Examples:
            >>> print(Affine([0, 1]) - Affine([3, 4]))
            -3.0 + -0.5e4 + 0.5e5
            >>> print(Affine([1, 2]) + 3)
            -1.5 + -0.5e6

        """
        if isinstance(other, self.__class__):
            x0 = self.x0 - other.x0
            xi = {}
            for i in self.xi:
                if i in other.xi:
                    val = fsub(self.xi[i], other.xi[i], rounding='u')
                    if val != 0:
                        xi[i] = val
                else:
                    xi[i] = self.xi[i]
            for i in other.xi:
                if i not in self.xi:
                    xi[i] = fneg(other.xi[i], rounding='u')
            return Affine(x0=x0, xi=xi)
        if isinstance(other, (int, float, mpmath.mpf)):
            x0 = self.x0 - mp.mpf(other)
            xi = self.xi.copy()
            return Affine(x0=x0, xi=xi)
        raise affapyError("other must be Affine, int, float, mpf")

    def __rsub__(self, other):
        """
        **Reverse operator -**

        Subtract two affine forms or an integer or
        float or mpf and an affine form.
        See the sub operator for more details.

        Args:
            self (Affine): second operand
            other (Affine or int or float or mpf): first operand

        Returns:
            Affine: other - self

        Raises:
            affapyError: other must be Affine, int, float, mpf

        Examples:
            >>> print(3 - Affine([1, 2]))
            1.5 + 0.5e1

        """
        return -self + other

    def __mul__(self, other):
        """
        **Operator ***

        Multiply two affine forms:

        .. math ::
            \\hat{x}\\hat{y} =
            x_0y_0 + \\sum_{i=1}^{n} (x_0y_i + y_0x_i)\\epsilon_i
            + rad(x)rad(y)\\epsilon_k

        :math:`k` is a new noise symbol.
        Or multiply an affine form and integer or float or mpf:

        .. math ::
            \\hat{x}y =
            x_0y + \\sum_{i=1}^{n} x_iy\\epsilon_i

        Args:
            self (Affine): first operand
            other (Affine or int or float or mpf): second operand

        Returns:
            Affine: self * other

        Raises:
            affapyError: other must be Affine, int, float, mpf

        Examples:
            >>> print(Affine([1, 2]) * Affine([3, 4]))
            5.25 + -1.75e2 + -0.75e3 + 0.25e4

        """
        if isinstance(other, self.__class__):
            x0 = self.x0 * other.x0
            xi = {}
            keyMax = max(max(self.xi) if self.xi else 0,
                         max(other.xi) if other.xi else 0)
            for i in range(keyMax + 1):
                v = 0
                if i in self.xi and i not in other.xi:
                    v = fmul(self.xi[i], other.x0, rounding='u')
                elif i not in self.xi and i in other.xi:
                    v = fmul(other.xi[i], self.x0, rounding='u')
                elif i in self.xi and i in other.xi:
                    v = fadd(fmul(self.xi[i], other.x0, rounding='u'),
                             fmul(other.xi[i], self.x0, rounding='u'),
                             rounding='u')
                if v != 0:
                    xi[i] = v
            xi[Affine._getNewXi()] = fmul(self.rad(),
                                          other.rad(), rounding='u')
            return Affine(x0=x0, xi=xi)
        if isinstance(other, (int, float, mpmath.mpf)):
            x0 = mp.mpf(other) * self.x0
            xi = {i: fmul(mp.mpf(other),
                          self.xi[i], rounding='u') for i in self.xi}
            return Affine(x0=x0, xi=xi)
        raise affapyError("other must be Affine, int, float, mpf")

    def __rmul__(self, other):
        """
        **Reverse operator ***

        Multiply two affine forms or an integer
        or float or mpf and an affine form.
        See the mul operator for more details.

        Args:
            self (Affine): second operand
            other (Affine or int or float or mpf): first operand

        Returns:
            Affine: other * self

        Raises:
            affapyError: other must be Affine, int, float, mpf

        """
        return self * other

    # Non-affine operations
    def _affineConstructor(self, alpha, dzeta, delta):
        """
        **Affine constructor**

        Return the affine form for non-affine operations:

        .. math ::
            \\hat{\\chi} =
            (\\alpha x_0 + \\zeta) + \\sum_{i=1}^{n} \\alpha x_i\\epsilon_i
            + \\delta \\epsilon_k

        :math:`k` is a new noise symbol.

        Args:
            alpha (mpmath.mpf)
            dzeta (mpmath.mpf)
            delta (mpmath.mpf)

        Returns:
            Affine: construction of an affine form

        """
        x0 = alpha * self.x0 + dzeta
        xi = {i: fmul(alpha, self.xi[i], rounding='u') for i in self.xi}
        xi[Affine._getNewXi()] = delta
        return Affine(x0=x0, xi=xi)

    def inv(self):
        """
        **Inverse**

        Return the inverse of an affine form.
        It uses the affine constructor with:

        .. math ::
            \\alpha = -\\frac{1}{b^2}

        .. math ::
            \\zeta = abs(mid(i)),

        .. math ::
            \\delta = radius(i)

        with:

        .. math ::
            i = [\\frac{1}{a} - \\alpha a, \\frac{2}{b}]

        .. math ::
            a = min(|inf|, |sup|)

        .. math ::
            b = max(|inf|, |sup|)

        where :math:`[inf, sup]` is the interval associated to the affine
        form in argument.

        Args:
            self: operand

        Returns:
            Affine: 1 / self

        Raises:
            affapyError: the interval associated to the affine form contains 0

        """
        if 0 not in self.interval:
            inf, sup = self.interval.inf, self.interval.sup
            a, b = min(fabs(inf), fabs(sup)), max(fabs(inf), fabs(sup))
            alpha = -1 / b**2
            i = affapy.ia.Interval(1/a - alpha*a, 2/b)
            dzeta = i.mid()
            if inf < 0:
                dzeta = -dzeta
            delta = i.radius()
            return self._affineConstructor(alpha, dzeta, delta)
        raise affapyError(
            "the interval associated to the affine form contains 0")

    def __truediv__(self, other):
        """
        **Operator /**

        Divide two affine forms or an integer or float or mpf
        and an affine form. It uses the identity:

        .. math ::
            \\frac{x}{y} = x \\times \\frac{1}{y}

        Args:
            self (Affine): first operand
            other (Affine or int or float or mpf): second operand

        Returns:
            Affine: self / other

        Raises:
            affapyError: other must be Affine, int, float, mpf

        Examples:
            >>> print(Affine([1, 2]) / Affine([3, 4]))
            0.4375 + -0.145833333333333e11 + 0.046875e12 +
            0.0156249999999999e13 + 0.0208333333333333e14

        """
        if isinstance(other, self.__class__):
            return self * other.inv()
        if isinstance(other, (int, float, mpmath.mpf)):
            return self * (1 / other)
        raise affapyError("other must be Affine, int, float, mpf")

    def __rtruediv__(self, other):
        """
        **Reverse operator /**

        Divide two affine forms or an affine form and an integer
        or float or mpf. See the truediv operator for more details.

        Args:
            self (Affine): second operand
            other (Affine or int or float or mpf): first operand

        Returns:
            Affine: other / self

        Raises:
            affapyError: other must be Affine, int, float, mpf

        Examples:
            >>> print(2 / Affine([1, 2]))
            1.5 + 0.25e15 + 0.25e16

        """
        if (isinstance(other, self.__class__) or
                isinstance(other, (int, float, mpmath.mpf))):
            return other * self.inv()
        raise affapyError("other must be Affine, int, float, mpf")

    def sqr(self):
        """
        Return the square of an affine form.
        It uses the identity:

        .. math ::
            x^2 = x \\times x

        Args:
            self (Affine): operand

        Returns:
            Affine: self ** 2

        """
        return self * self

    def __pow__(self, n):
        """
        **Operator ****

        Return the power of an affine form with another affine form
        or an integer.
        With an affine, it uses the identity:

        .. math ::
            x^n = exp(n \\times log(x))

        Args:
            self (Affine): first operand
            n (Affine or int): second operand (exponent)

        Returns:
            Affine: self ** n

        Raises:
            affapyError: type error: n must be Affine or int

        Examples:
            >>> print(Affine([1, 2])**3)
            3.375 + -3.375e17 + 0.375e18 + 0.875e19

        """
        if isinstance(n, int):
        
            if n < 0:
                x = self.inv()
                n = -n
            if n == 0:
                return Affine([1, 1])
            y = 1
            x = self.copy()
            while n > 1:
                if n % 2 == 0:
                    x = x * x
                    n = n / 2
                else:
                    y = x * y
                    x = x * x
                    n = (n - 1) / 2
            return x * y
        else:
            int(n)
            
            
        if isinstance(n, self.__class__) or isinstance(n, float): 
            if n < 0: 
                self = self.inv()
                n = -n
            if 0 in self.interval:
                if float(n).is_integer():
                    a = min(self.interval.inf**n, 0.0)
                    b = max(self.interval.inf**n, self.interval.sup**n)
                    return Affine([a, b])   
                raise affapyError("ValueError: Negative base with irrational exponent")
                
            if self.interval > 0 or (self.interval < 0 and float(n).is_integer() and n%2 !=0): 
                a = self.interval.inf
                b = self.interval.sup
                if self._approxMethod == 'minRange':
                    alpha = n * a**(n-1)          
                    dzeta, delta = self.approx_minRange(a**n, b**n, alpha, a, b)                
                    return self._affineConstructor(alpha, dzeta, delta)
                
                if self._approxMethod == 'minCheb':
                    alpha = fdiv(fsub(b**n, a**n), fsub(b, a))
                    u = (alpha)**(1/(n-1.0)) / float(n)
                    dzeta, delta = self.approx_cheb(exp(a), exp(b), alpha / float(n), a, b, u)             
                    return self._affineConstructor(alpha, dzeta, delta)        
                
                else: return (n * self.log()).exp()
                
            if self.interval < 0 and n % 2 == 0: 
                a = self.interval.inf
                b = self.interval.sup
                
                if self._approxMethod == 'minRange':
                    alpha = n * b**(n-1)          
                    dzeta, delta = self.approx_minRange(b**n, a**n, alpha, b, a)                
                    return self._affineConstructor(alpha, dzeta, delta)
                
                if self._approxMethod == 'minCheb':
                    alpha = fdiv(fsub(b**n, a**n), fsub(b, a))
                    u = (alpha)**(1/(n-1.0)) / float(n)
                    dzeta, delta = self.approx_cheb(exp(a), exp(b), alpha / float(n), a, b, u)             
                    return self._affineConstructor(alpha, dzeta, delta)                
                
                else: return (n * abs(self).log()).exp() 
                
            raise affapyError("ValueError: Negative base with irrational exponent")
   
        raise affapyError("type error: n must be Affine, float or int")

    # Functions
    def __abs__(self):
        """
        Return the absolute value of an affine form.
        Three possibilities:

        1. If :math:`x < 0`:

        .. math ::
            |\\hat{x}| = -\\hat{x}

        2. If :math:`x > 0`:

        .. math ::
            |\\hat{x}| = \\hat{x}

        3 If :math:`x` straddles :math:`0`:

        .. math ::
            |\\hat{x}| = \\frac{|x_0|}{2} +
            \\sum_{i=1}^{n} \\frac{x_i\\epsilon_i}{2}

        Args:
            self (Affine): operand

        Returns:
            Affine: abs(self)

        Examples:
            >>> print(abs(Affine([1, 2])))
            1.5 + -0.5e24
            >>> print(abs(Affine([-2, -1])))
            1.5 + 0.5e25

        """
        if self.strictly_neg():
            return -self
        if self.straddles_zero():
            x0 = fabs(self.x0 / 2)
            xi = {i: fdiv(self.xi[i], 2, rounding='u') for i in self.xi}
            return Affine(x0=x0, xi=xi)
        return self.copy()

    def sqrt(self):
        """
        **Function sqrt**

        Return the square root of an affine form.
        We consider the interval :math:`[a, b]` associated to the affine form.
        It uses the affine constructor with:

        .. math ::
            \\alpha = \\frac{1}{\\sqrt{b} + \\sqrt{a}}

        .. math ::
            \\zeta = \\frac{\\sqrt{a} + \\sqrt{b}}{8} + \\frac{1}{2}
            \\frac{\\sqrt{a}\\sqrt{b}}{\\sqrt{a} + \\sqrt{b}}

        .. math ::
            \\delta = \\frac{1}{8}\\frac{(\\sqrt{b}
            - \\sqrt{a})^2}{\\sqrt{a} + \\sqrt{b}}

        Args:
            self (Affine): operand

        Returns:
            Affine: sqrt(self)

        Raises:
            affapyError: the interval associated to the affine form must be >=0

        Examples:
            >>> print(Affine([1, 2]).sqrt())
            1.21599025766973 + -0.207106781186548e27 + 0.00888347648318441e28

        """
        if self.interval >= 0:
            a, b = self.interval.inf, self.interval.sup
            t = fadd(sqrt(a), sqrt(b), rounding='f')
            alpha = 1 / t
            dzeta = fadd(fdiv(t, 8), fmul(0.5, fdiv(sqrt(fmul(a, b)), t)))
            rdelta = fsub(sqrt(b), sqrt(a), rounding='u')
            delta = fdiv(fmul(rdelta, rdelta, rounding='u'),
                         fmul(8, t, rounding='f'), rounding='u')
            return self._affineConstructor(alpha, dzeta, delta)
        raise affapyError(
            "the interval associated to the affine form must be >= 0")

    def exp(self):
        """
        **Function exp**

        Return the exponential of an affine form.
        We consider the interval :math:`[a, b]` associated to the affine form.
        It uses the affine constructor with:

        .. math ::
            \\alpha = \\frac{exp(b) - exp(a)}{b - a}

        .. math ::
            \\zeta = \\alpha \\times (1 - log(\\alpha))

        .. math ::
            \\delta = \\frac{\\alpha \\times (log(\\alpha)-1-a)) + exp(a)}{2}

        Args:
            self (Affine): operand

        Returns:
            Affine: exp(self)

        Examples:
            >>> print(Affine([1, 2]).exp())
            4.47775520281461 + -2.3353871352358e29 + 4.95873115091173e30

        """
        
        
        a, b = self.interval.inf, self.interval.sup

        if self._approxMethod == 'minRange':
            alpha = exp(a)            
            dzeta, delta = self.approx_minRange(exp(a), exp(b), alpha, a, b)

            
        elif self._approxMethod == 'minCheb': 
            alpha = fdiv(fsub(exp(b), exp(a)), fsub(b, a))
            xs = log(alpha)
            dzeta, delta = self.approx_cheb(exp(a), exp(b), alpha, a, b, xs)
                                            
        else:
            xs = log(alpha)
            alpha = fdiv(fsub(exp(b), exp(a)), fsub(b, a))
            maxdelta = fadd(fmul(alpha, fsub(xs, fsub(1, a))), exp(a))
            dzeta = fmul(alpha, fsub(1, xs))
            delta = fdiv(maxdelta, 2)
        return self._affineConstructor(alpha, dzeta, delta)

    def log(self):
        """
        **Function log**

        Return the logarithm of an affine form.
        We consider the interval :math:`[a, b]` associated to the affine form.
        It uses the affine constructor with:

        .. math ::
            \\alpha = \\frac{log(b) - log(a)}{b - a}

        .. math ::
            \\zeta = \\frac{-\\alpha x_s}{\\frac{log(x_s) + y_s}{2}}

        .. math ::
            \\delta = \\frac{log(x_s) - y_s}{2}

        with:

        .. math ::
            x_s = \\frac{1}{\\alpha}

        .. math ::
            y_s = \\alpha (x_s - a) + log(a)

        Args:
            self (Affine): operand

        Returns:
            Affine: log(self)

        Raises:
            affapyError: the interval associated to the affine form must be > 0

        Examples:
            >>> print(Affine([1, 2]).log())
            -1.93043330907435 + -0.346573590279973e31 + 0.0298300505708048e32

        """
        if self.interval > 0:
            a, b = self.interval.inf, self.interval.sup
            la, lb = log(a), log(b)
            if self._approxMethod == 'minRange':
                alpha = fdiv(1.0, b)
                dzeta = fdiv(fsub(log(fmul(a,b)), fadd(fdiv(a,b), 1)), 2.0)
                delta = fdiv(fsub(log(fdiv(a,b)), fsub(fdiv(a,b), 1)), 2.0)
                
            elif self._approxMethod == 'minCheb':
                alpha = fdiv(fsub(lb, la), fsub(b, a))
                dzeta = fsub(log(fdiv(a,alpha)), fadd(1, fdiv(1,fmul(a,alpha))))
                delta = fsub(log(fdiv(1,fmul(a,alpha))), fsub(1,fmul(a,alpha)))
                            
            else:
                alpha = fdiv(fsub(lb, la), fsub(b, a))
                xs = fdiv(1, alpha)
                ys = fadd(fmul(alpha, fsub(xs, a)), la)
                maxdelta = fsub(log(xs), ys)
                dzeta = fdiv(fmul(alpha, fneg(xs)), fdiv(fadd(log(xs), ys), 2.0))
                delta = fdiv(maxdelta, 2.0)
                
            return self._affineConstructor(alpha, dzeta, delta)
        raise affapyError(
            "the interval associated to the affine form must be > 0")

    # Trigo
    def sin(self, npts=8):
        """
        **Function sin**

        Return the sinus of an affine form.
        It uses the least squares and the affine constructor.
        The argument npts is the number of points for the linear
        regression approximation.

        Args:
            self (Affine): operand
            npts (int): number of points (default: 8)

        Returns:
            Affine: sin(self)

        Examples:
            >>> print(Affine([1, 2]).sin())
            0.944892253579443 + -0.0342679845626557e33 + 0.0698628113164167e34


        """
        w = self.interval.width()
        a, b = self.interval.inf, self.interval.sup
        if w >= 2 * mp.pi:
            return Affine(interval=[-1, 1])
        # Case of the least squares
        x, y = [a], [sin(a)]
        pas = w / (npts - 1)
        for i in range(1, npts - 1):
            x.append(x[i - 1] + pas)
            y.append(sin(x[i]))
        x.append(b)
        y.append(sin(b))
        # Calculation of xm and ym, averages of x and y
        xm, ym = fsum(x) / npts, fsum(y) / npts
        # Calculation of alpha and dzeta
        temp2 = 0
        alpha = 0
        for xi, yi in zip(x, y):
            temp1 = xi - xm
            alpha += yi * temp1
            temp2 += temp1 * temp1
        alpha = alpha / temp2
        dzeta = ym - alpha * xm
        # Calculation of the residues
        r = [fabs(yi - (dzeta + alpha * xi)) for xi, yi in zip(x, y)]
        # The error delta is the maximum of the residues (in absolute values)
        delta = max(r)
        return self._affineConstructor(alpha, dzeta, delta)

    def cos(self):
        """
        **Function cos**

        Return the cosinus of an affine form.
        It uses the identity:

        .. math ::
            cos(x) = sin\\left(x + \\frac{\\pi}{2}\\right)

        Args:
            self (Affine): operand

        Returns:
            Affine: cos(self)

        """
        return (self + mp.pi / 2).sin()

    def tan(self):
        """
        **Function tan**

        Return the tangent of an affine form.
        It uses the identity:

        .. math ::
            tan(x) = \\frac{sin(x)}{cos(x)}

        Args:
            self (Affine): operand

        Returns:
            Affine: tan(self)

        """
        return self.sin() / self.cos()

    def cotan(self):
        """
        **Function cotan**

        Return the cotangent of an affine form.
        It uses the identity:

        .. math ::
            cotan(x) = \\frac{cos(x)}{sin(x)}

        Args:
            self (Affine): operand

        Returns:
            Affine: cotan(self)

        """
        return self.cos() / self.sin()

    def cosh(self):
        """
        **Function cosh**

        Return the hyperbolic cosine of an affine form.
        It uses the identity:

        .. math ::
            cosh(x) = \\frac{exp(x) + exp(-x)}{2}

        Args:
            self (Affine): operand

        Returns:
            Affine: cosh(self)

        """
        return (self.exp() + (-self).exp()) * 0.5

    def sinh(self):
        """
        **Function sinh**

        Return the hyperbolic sine of an affine form.
        It uses the identity:

        .. math ::
            sinh(x) = \\frac{exp(x) - exp(-x)}{2}

        Args:
            self (Affine): operand

        Returns:
            Affine: sinh(self)

        """
        return (self.exp() - (-self).exp()) * 0.5

    def tanh(self):
        """
        **Function tanh**

        Return the hyperbolic tangeant of an affine form.
        It uses the identity:

        .. math ::
            tanh(x) = \\frac{sinh(x)}{cosh(x)}

        Args:
            self (Affine): operand

        Returns:
            Affine: tanh(self)

        """
        return self.sinh() / self.cosh()

    # Comparison operators
    def __eq__(self, other):
        """
        **Operator ==**

        Compare two affine forms.

        Args:
            self (Affine): first operand
            other (Affine): second operand

        Returns:
            bool: self == other

        Raises:
            affapyError: other must be Affine

        """
        if isinstance(other, self.__class__):
            return self.x0 == other.x0 and self.xi == other.xi
        raise affapyError("other must be Affine")

    def __ne__(self, other):
        """
        **Operator !=**

        Negative comparison of two affine forms.

        Args:
            self (Affine): first operand
            other (Affine): second operand

        Returns:
            bool: self != other

        Raises:
            affapyError: other must be Affine

        """
        if isinstance(other, self.__class__):
            return self.x0 != other.x0 or self.xi != other.xi
        raise affapyError("other must be Affine")

    # Inclusion
    def __contains__(self, other):
        """
        **Operator in**

        Return True if the interval of self is in the interval of other.

        Args:
            self (Affine): first operand
            other (Affine): second operand

        Returns:
            bool: self in other

        Raises:
            affapyError: other must be Affine, Interval, int, float, mpf

        """
        if isinstance(other, self.__class__):
            return other.interval in self.interval
        if isinstance(other, affapy.ia.Interval):
            return other in self.interval
        if isinstance(other, (int, float, mpmath.mpf)):
            return other in self.interval
        raise affapyError("other must be Affine, Interval, int, float, mpf")

    def straddles_zero(self):
        """
        Return True if the affine form straddles 0, False if not.

        Args:
            self (Affine): operand

        Returns:
            bool: 0 in self

        """
        return self.interval.straddles_zero()

    def strictly_neg(self):
        """
        Return True if the affine is strictly negative, False if not.

        Args:
            self (Affine): operand

        Returns:
            bool: self < 0

        """
        return self.interval < 0

    # Formats
    def __str__(self):
        """
        **String format**

        Make the string format.

        Args:
            self (Affine): arg

        Returns:
            string: sum of noise symbols

        Examples:
            >>> print(Affine([1, 2]))
            1.5 - 0.5*e1

        """
        return " + ".join(
            [str(self.x0)] +
            ["".join([str(self.xi[i]), "e", str(i)]) for i in self.xi])

    def __repr__(self):
        """
        **Repr format**

        Make the repr format.

        Args:
            self (Affine): arg

        Returns:
            string: format

        """
        return "Affine({}, {})".format(self.x0, self.xi)

    def copy(self):
        """
        Copy an affine form.

        Args:
            self (Affine): arg

        Returns:
            Affine: self copy

        """
        return Affine(x0=self.x0, xi=self.xi.copy())

    def convert(self):
        """
        Convert an affine form to an interval representation:

        .. math ::
            X = [x_0 + rad(x), x_0 - rad(x)]

        with:

        .. math ::
            rad(x) = \\sum_{i=1}^{n} |x_i|

        Args:
            self (Affine): arg

        Returns:
            Interval: interval associated to the affine form

        """
        return self.interval.copy()


    def general_approx(self, alpha, u, ru, fu):
        zeta = fsub(fdiv(fadd(fu, ru), 2.0), fmul(alpha, u))
        delta = fdiv(abs(fsub(fu, ru)), 2.0)
        return zeta, delta
    
    
    def approx_cheb(self, fa, fb, fu, a, b, u):
        alpha = (fb - fa) / (b - a)
        ru = fa + alpha * (u - a)
        return self.general_approx(alpha, u, ru, fu)
    
        
    def approx_minRange(self, fp, fu, alpha, p, u):
        ru = fp + alpha * (u - p)
        return self.general_approx(alpha, u, ru, fu)        
        
    def mpi2aff(x):
        return Affine([mpmath.mpf(x.a), mpmath.mpf(x.b)])

    def aff2mpi(x):
        return mpmath.mpi(x.interval.inf, x.interval.sup) 
    
    def affList2mpiList(x):
        return [Affine.aff2mpi(xi) for xi in x]

    def mpiList2affList(x):
        return [Affine.mpi2aff(xi) for xi in x]            