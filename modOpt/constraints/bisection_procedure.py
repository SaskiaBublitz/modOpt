import numpy
import sympy
import mpmath
import copy
import tables

class Particle(tables.IsDescription):
	boxcombination = tables.StringCol(1e6)

def bisectionReduction(model, noReductionSteps, boxSelectionMethod):

	bounds = copy.deepcopy(model.xBounds)
	
	for rs in range(noReductionSteps):
		print('reduction Step No: '+str(rs))
		results = []
		for b in range(len(bounds)):
			reducedBoxes = cutConstraintsInHalfWithBisection(bounds[b],
								model.fSymbolic, model.xSymbolic, boxSelectionMethod)

			if reducedBoxes != []:
				for r in reducedBoxes:
					results.append(r)

		bounds = results

	return bounds

def cutConstraintsInHalfWithBisection(Bounds, fSymbolic, xSymbolic, selectionMethod):

	reducedBoxes = []

	subBoxList = splittBoxInMidpoint(Bounds)
	#subBoxes = numpy.load('boxComData.npy', mmap_mode='r')
	print(subBoxList)
	BoxCombAlgorithmus(subBoxList)

	return []

	Notes = determineNotesOfBox(Bounds)

	if checkForSignChange(Notes, fSymbolic, xSymbolic) == False:
		print('There seems to be no single solution in this Intervals')
		return []

	else:
		subBoxes = splittBoxInMidpoint(Bounds)
		print('new subBox')
		for s in subBoxes:
			notes = determineNotesOfBox(s)
			if checkForSignChange(notes, fSymbolic, xSymbolic) == True:
				if selectionMethod == 'save':
					reducedBoxes.append(s)
				elif selectionMethod == 'secant':
					if checkBoxWithSecant(notes, fSymbolic, xSymbolic) == True:
						reducedBoxes.append(s)

	return reducedBoxes


def determineNotesOfBox(Bounds):
	'''returns array of Note arrays (coordinates of Box corners)'''

	boundslist = numpy.zeros((len(Bounds), 2), dtype=numpy.float32)
	for b in range(len(Bounds)):
		boundslist[b][0] = mpmath.mpf(Bounds[b].a)
		boundslist[b][1] = mpmath.mpf(Bounds[b].b)
	
	Notes = numpy.array(numpy.meshgrid(*boundslist)).T.reshape(-1,(len(Bounds)))

	return Notes.astype(numpy.float32)


def checkForSignChange(Notes, fSymbolic, xSymbolic):
	fLamb = sympy.lambdify(xSymbolic, fSymbolic, 'numpy')
	
	notes_res = numpy.zeros((len(Notes), len(fSymbolic)))
	for n, note in enumerate(Notes):
		notes_res[n] = fLamb(*note)

	signChange = []
	for s in range(len(fSymbolic)):
		if all(i>=0 for i in notes_res[:,s]) or all(i<=0 for i in notes_res[:,s]):
			signChange.append(False)
		else:
			signChange.append(True)

	return numpy.all(signChange)


def splittBoxInMidpoint(Bounds):

	ivSubBoxList = []
	for bound in Bounds:
		ivSubBoxList.append([mpmath.mpi(bound.a, bound.mid), mpmath.mpi(bound.mid, bound.b)])

	return ivSubBoxList
	#numpy.save('boxComData.npy',numpy.array(numpy.meshgrid(*ivSubBoxList)).T.reshape(-1,(len(Bounds))))


def checkBoxWithSecant(Notes, fSymbolic, xSymbolic):
	'''noch in Arbeit'''
	fLamb = sympy.lambdify(xSymbolic, fSymbolic, 'numpy')
	
	notes_res = numpy.zeros((len(Notes), len(fSymbolic)))
	for n, note in enumerate(Notes):
		notes_res[n] = fLamb(*note)

	interpolatedRoots = []
	for s in range(len(fSymbolic)):
		posSign = []
		negSign = []
		for n, note_res in enumerate(notes_res[:,s]):
			if note_res >= 0: posSign.append(Notes[n])
			else: negSign.append(Notes[n])

		Lines = []
		for p in posSign:
			for n in negSign:
				Lines.append([p,n])

		for l in Lines:
			iroot = l[1]-fLamb(*l[1])[s]*((l[1]-l[0])/(fLamb(*l[1])[s]-fLamb(*l[0])[s]))
			interpolatedRoots.append(iroot)
		
	functionvaluesOfInterpolatedPoints = []
	for i in interpolatedRoots:
		functionvaluesOfInterpolatedPoints.append(fLamb(*i))
	functionvaluesOfInterpolatedPoints = numpy.array(functionvaluesOfInterpolatedPoints)
	print(functionvaluesOfInterpolatedPoints)	

	signChange = []
	for s in range(len(fSymbolic)):
		if (all(i>=0 for i in functionvaluesOfInterpolatedPoints[:,s]) or
			 all(i<=0 for i in functionvaluesOfInterpolatedPoints[:,s])):
			signChange.append(False)
		else:
			signChange.append(True)

	print(numpy.all(signChange))
	return numpy.all(signChange)


def BoxCombAlgorithmus(splittedBounds):

	h5file = tables.open_file('BisectionData.h5', mode='w', title='Stored data for computation')
	group = h5file.create_group("/", 'bisectionGroup', 'bisection Data Group')
	table = h5file.create_table(group, 'BoxComb', Particle, "Box combinations")

	lenBounds = len(splittedBounds)
	possibleKombinations = 2**lenBounds

	repeatingArray = numpy.zeros(lenBounds)
	for r in range(len(repeatingArray)):
		repeatingArray[r] = 2**(lenBounds-1-r)

	kombination = numpy.zeros(lenBounds, dtype=object)
	currentrepeating = numpy.zeros(lenBounds)
	for k in range(possibleKombinations):
		print(k)
		for s in range(len(splittedBounds)):
			if currentrepeating[s] < repeatingArray[s]:
				kombination[s] = splittedBounds[s][0]
				currentrepeating[s] = currentrepeating[s]+1
			elif currentrepeating[s] < 2*repeatingArray[s]:
				kombination[s] = splittedBounds[s][1]
				currentrepeating[s] = currentrepeating[s]+1
			else:
				kombination[s] = splittedBounds[s][0]
				currentrepeating[s] = 1

		box = table.row
		box['boxcombination'] = mpmath.nstr(kombination)
		box.append()
		table.flush()


	h5file.close()


def splittBoxForDefinedIntervals(Bounds, splittBooleanList):
			
	ivSubBoxList = []
	for b, bound in enumerate(Bounds):
		if splittBooleanList[b] == True:
			ivSubBoxList.append([mpmath.mpi(bound.a, bound.mid), mpmath.mpi(bound.mid, bound.b)])
		else:
			ivSubBoxList.append([bound])

	SubBoxes = numpy.array(numpy.meshgrid(*ivSubBoxList)).T.reshape(-1,(len(Bounds)))

	SubBoxList = []
	for s in SubBoxes:
		SubBoxList.append(s)


	return SubBoxList
