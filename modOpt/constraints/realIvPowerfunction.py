'''
This script overrides the __pow__ (**) function of an ivmpf object, if importet.
Should be importet after mpmath is importet
'''
import mpmath
from mpmath.ctx_iv import ivmpf
from fractions import Fraction


#function to calculate roots of negative and positive numbers
def realRoot(x, r):
	if x >= 0: # root of positive number
		return mpmath.root(x, r)
	else:
		if r % 2 != 0: # odd root
			return -mpmath.root(-x, r)
		else: # even root of negative number
			return mpmath.mpi(float('nan'))


# defining generel iv-power function that returns only real intervals
def ivRealPower():

	def pow(base, power):
		if isinstance(power, mpmath.ctx_iv.ivmpf):
			lowerBoundPower = floatPow(base, float(mpmath.convert(power.a)))
			upperBoundPower = floatPow(base, float(mpmath.convert(power.b)))
			lowerBound = min(lowerBoundPower.a, upperBoundPower.a)
			upperBound = max(lowerBoundPower.b, upperBoundPower.b)
			return mpmath.mpi(lowerBound, upperBound)
        
		else:
			return floatPow(base, power)


	def floatPow(base, power):
		if power > 0:
			return posPow(base, power)
		elif power == 0:
			return mpmath.mpi(1.0, 1.0)
		else:
			return 1/posPow(base, -power)


	def posPow(base, power):

		if float(power).is_integer(): # whole number
			if power % 2 == 0: # power even
				if 0 <= base.a <= base.b:
					return mpmath.mpi(mpmath.power(base.a, power), mpmath.power(base.b, power))
				elif base.a <= base.b <= 0:
					return mpmath.mpi(mpmath.power(base.b, power), mpmath.power(base.a, power))
				else:
					return mpmath.mpi(0, max(mpmath.power(base.a, power), mpmath.power(base.b, power)))
			else: # power odd
				return mpmath.mpi(mpmath.power(base.a, power), mpmath.power(base.b, power))
		else: # not whole number
			frac = Fraction(power).limit_denominator()
			if frac.denominator % 2 != 0: # denominator odd
				lowBoundPow = realRoot(mpmath.power(base.a, frac.numerator), frac.denominator)
				upBoundPow = realRoot(mpmath.power(base.b, frac.numerator), frac.denominator)
				if frac.numerator % 2 == 0: # numerator even
					if 0 <= base.a <= base.b:
						return mpmath.mpi(lowBoundPow, upBoundPow)
					elif base.a <= base.b <= 0:
						return mpmath.mpi(upBoundPow, lowBoundPow)
					else:
						return mpmath.mpi(0, max(upBoundPow, lowBoundPow))
				else: # numerator also odd
						return mpmath.mpi(lowBoundPow, upBoundPow)
			else: # dnominator even
				if frac.numerator % 2 == 0: # numerator also even
					lowBoundPow = realRoot(mpmath.power(base.a, frac.numerator), frac.denominator)
					upBoundPow = realRoot(mpmath.power(base.b, frac.numerator), frac.denominator)
					if 0 <= base.a <= base.b:
						return mpmath.mpi(lowBoundPow, upBoundPow)
					elif base.a <= base.b <= 0:
						return mpmath.mpi(upBoundPow, lowBoundPow)
					else:
						return mpmath.mpi(0, max(upBoundPow, lowBoundPow))
				else: # numerator odd
					lowBoundPow = realRoot(mpmath.power(base.a, frac.numerator), frac.denominator)
					upBoundPow = realRoot(mpmath.power(base.b, frac.numerator), frac.denominator)
					if 0 <= base.a <= base.b:
						return mpmath.mpi(lowBoundPow, upBoundPow)
					elif base.a <= base.b <= 0:
						return mpmath.mpi(float('nan'), float('nan'))
					else: 
						return mpmath.mpi(0, upBoundPow)

	return pow


#overriding power function in mpmath package
ivmpf.__pow__ = ivRealPower()


# Test
#print(mpmath.mpi(-2,-1.5)**(-2/3))

