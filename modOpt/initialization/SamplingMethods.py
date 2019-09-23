'''
Created on Aug 17, 2018

@author: e.esche
'''

from pyDOE import lhs # might require installation: pip install pyDOE
# from prime import list_prime
from sympy import sieve

import numpy
from numpy import zeros, ones
from numpy.random import normal
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt

# TODO:

SamplingMethodList = ['EV', 'LHS', 'HSS', 'NDS','LHS_NDS','LHS_gamma','value']
ControlSamplingMethodList = ['NDS', 'LHS_NDS', 'LHS_gamma']

class SamplingMethods:
    '''
    classdocs
    '''
    def __init__(self, samplingMethod, numberOfSamples=1, inputSpace=[]):
        '''
        Constructor
        '''
        if not inputSpace.performSampling:
            samplingMethod = 'value'
            print 'Sampling deactivated: expected values used'

        self.samples = []
        if samplingMethod in SamplingMethodList:
            print "Selected sampling method: ", samplingMethod
            
            if samplingMethod == 'LHS':
                lh_data = lhs(inputSpace.globalID+1, numberOfSamples, 'center')
                currentSamples = lh_data
                for i in range(0,inputSpace.globalID+1):
                    currentSamples[:,i] = currentSamples[:,i] * (inputSpace.getUpperBound(i) - inputSpace.getLowerBound(i)) + inputSpace.getLowerBound(i)
                self.samples = currentSamples
            
            elif samplingMethod == 'EV':
                currentSamples = zeros((1, inputSpace.globalID+1))
                for i in range(0,inputSpace.globalID+1):
                    currentSamples[:,i] = 0.5 * (inputSpace.getUpperBound(i) - inputSpace.getLowerBound(i)) + inputSpace.getLowerBound(i)
                self.samples = currentSamples
            
            elif samplingMethod == 'NDS':
                stdv = zeros((inputSpace.globalID+1))
                mean = zeros((inputSpace.globalID+1))
                for i in range(0,inputSpace.globalID+1):
                    stdv[i] = inputSpace.getUpperBound(i) - inputSpace.getLowerBound(i)
                    mean[i] = inputSpace.getLowerBound(i) + 0.5*stdv[i]
                    stdv[i] = stdv[i]/6.0
                currentSamples = zeros((numberOfSamples, inputSpace.globalID+1))
                for i in range(0,numberOfSamples):
                    for j in range(0,inputSpace.globalID+1):
                        nd_data = normal(loc=mean[j], scale=stdv[j], size=1)
                        currentSamples[i,j] = nd_data
                self.samples = currentSamples
            
            elif samplingMethod == 'HSS':
                self.samples = self.generateHammersleySamples(inputSpace, numberOfSamples)
            
            elif samplingMethod == 'LHS_NDS':
                stdv = zeros((inputSpace.globalID+1))
                mean = zeros((inputSpace.globalID+1))
                for i in range(0,inputSpace.globalID+1):
                    stdv[i] = inputSpace.getUpperBound(i) - inputSpace.getLowerBound(i)
                    mean[i] = inputSpace.getLowerBound(i) + 0.5*stdv[i]
                    stdv[i] = stdv[i]/6.0
                lh_data = lhs(inputSpace.globalID+1, numberOfSamples, 'center')
                currentSamples = lh_data
                for i in range(0,inputSpace.globalID+1):
                    currentSamples[:,i] = norm(loc=mean[i], scale=stdv[i]).ppf(currentSamples[:,i])
                self.samples = currentSamples
            
            elif samplingMethod == 'LHS_gamma':
                # If ForwarDist is True: the maximum of the gamma distribution will lay near the lower bound
                # If ForwarDist is False: the maximum of the gamma distribution will lay near the upper bound
                forwardDist = True
                # This parameter changes the skewness of the distribution
                a = 10.0

                deltaX = gamma.ppf(0.999, a, loc=0, scale=1)-gamma.ppf(0.001, a, loc=0, scale=1)
                loc = zeros((inputSpace.globalID+1))
                scale = zeros((inputSpace.globalID+1))
                for i in range(0,inputSpace.globalID+1):
                    loc[i] = inputSpace.getLowerBound(i)
                    scale[i] = (inputSpace.getUpperBound(i)-inputSpace.getLowerBound(i))/deltaX
                lh_data = lhs(inputSpace.globalID+1, numberOfSamples, 'center')
                currentSamples = lh_data
                for i in range(0,inputSpace.globalID+1):
                    currentSamples[:,i] = gamma.ppf(currentSamples[:,i], a, loc[i], scale[i])
                if forwardDist:
                    self.samples = currentSamples
                else:
                    for i in range(0,inputSpace.globalID+1):
                        currentSamples[:,i] = ones((numberOfSamples))*inputSpace.getUpperBound(i) - (currentSamples[:,i] - ones((numberOfSamples))*inputSpace.getLowerBound(i))
                    self.samples = currentSamples
            
            elif samplingMethod == 'value':
                currentSamples = []
                for i in range(inputSpace.globalID + 1):
                    currentSamples.append(inputSpace.getValues(inputSpace.varlist[i].varName))
                self.samples = currentSamples

        else:
            print "ERROR: Select a valid sampling method!"
            exit()
            
    def getSamples(self):
        return self.samples
    
    def plotSamples(self, sample0=0, sample1=1, plotHistogram=False):

        if len(self.samples[0]) > 1:
            plt.plot(self.samples[:,sample0], self.samples[:,sample1],'*')
            plt.show()

            if plotHistogram:
                plt.hist(self.samples[:,sample0])
                plt.show()
        
        else:
            plt.hist(self.samples[:,0])
            plt.show()


    def generateHammersleySamples(self, inputSpace, numberOfSamples):
        ''' bound0 the boundaries for all sampling points, where the number of tuples gives the number of dimensions D
        N is the number of sampling points
        '''

        D = inputSpace.globalID+1
        
        S = self.generateHammersleySequence(D, numberOfSamples)
            
        B = numpy.zeros(S.shape)
        
        for i in range(D):
            B[:,i] = S[:,i] * (inputSpace.varlist[i].upperBound - inputSpace.varlist[i].lowerBound) + inputSpace.varlist[i].lowerBound
        
        return B

    
    def generateHammersleySequence(self, D, N):
        ''' generate returns a set N points for a D-dimensional Hammersley sequence
        the interval (0,1)
        '''
        
        S = numpy.empty(shape=[N,D],dtype=float)
        
        '''last column is simple'''
        #S[:,-1] = numpy.arange(1.,N+1) / N - 1./(2*N)
        S[:,-1] = numpy.arange(0.,N) / N
        
        pn = 2.*D
        '''getting list of prime numbers'''
        # p = list_prime(2,int(pn))
        p = [i for i in sieve.primerange(2, int(pn))]
        
        while(len(p) < D):
            pn = 2 * pn
            # p = list_prime(2,int(pn))
            p = [i for i in sieve.primerange(2, int(pn))]
        
        P = p[0:D-1] # last dimension is already set
        '''loop for dimensions'''
        for k in range(0,D-1): 
            pk = P[k]
            '''loop for hammersley points'''
            for j in range(0,N):
                bj = j + 1
                '''maximum for the devision of binary logarithms'''
                n = numpy.int(numpy.max([1,numpy.round(numpy.log2(bj+1)/numpy.log2(pk))]))            
                while pk**n <= bj:
                    n = n + 1
                
                b = numpy.zeros(n)
                b[n-1] = bj % pk
                while bj and n > 1:
                    n = n - 1
                    bj = numpy.floor_divide(bj,pk)
                    b[n-1] = bj % pk
                
                S[j,k] = numpy.sum(b[::-1]/pk**numpy.arange(1.,len(b)+1))
        return S

class controlSamplingMethods:
    '''
    classdocs
    '''
    def __init__(self, time, inputSpace, samplingMethod='NDS', numberOfSamples=10):
        '''
        Constructor
        '''
        for i in range(inputSpace.globalID+1):
            if time['engUnit'] != inputSpace.varlist[i].time['engUnit']:
                print 'ERROR: time unit of control', inputSpace.varlist[i].varName, 'is not equal to unit of overall time unit'
                exit()
            if inputSpace.varlist[i].time['values'][0] > 0.0:
                print 'ERROR: No control value at time 0.0 in',inputSpace.varlist[i].varName
                exit()

        if not inputSpace.performSampling:
            print "No sampling for controls performed"
            self.sampleTime = []
            for i in range(inputSpace.globalID+1):
                self.sampleTime = numpy.concatenate((self.sampleTime, inputSpace.varlist[i].time['values']))
            self.sampleTime = {'values': numpy.unique(self.sampleTime), 'engUnit':time['engUnit']}
            self.time = {'values': numpy.unique(numpy.concatenate((time['values'], self.sampleTime['values']))), 'engUnit':time['engUnit']}
            
            self.controlSamples = zeros((inputSpace.globalID+1,len(self.sampleTime['values'])))
            for i in range(inputSpace.globalID+1):
                count=0
                self.controlSamples[i,0] = inputSpace.varlist[i].value[0]
                for j in range(1,len(self.sampleTime['values'])):
                    if self.sampleTime['values'][j] in inputSpace.varlist[i].time['values']:
                        count = count + 1
                    self.controlSamples[i,j] = inputSpace.varlist[i].value[count]
            return

        if samplingMethod in ControlSamplingMethodList:
            print "Selected control sampling method: ", samplingMethod
            
            self.sampleTime = {'values': numpy.linspace(time['values'][0],time['values'][-1],numberOfSamples+1), 'engUnit':time['engUnit']}
            self.time = {'values': numpy.unique(numpy.concatenate((time['values'], self.sampleTime['values']))), 'engUnit':time['engUnit']}

            firstStageSampling = numpy.random.randint(2, size=(inputSpace.globalID+1, numberOfSamples))
            firstSamples = numpy.ones((inputSpace.globalID+1,1), dtype='int64')
            firstStageSampling = numpy.concatenate((firstSamples, firstStageSampling), axis=1)
            numberOfControlActions = numpy.sum(firstStageSampling,axis=1)

            secondStageSampling = []
            
            if samplingMethod == 'NDS':
                for i in range(0,inputSpace.globalID+1):
                    stdv = inputSpace.getUpperBound(i) - inputSpace.getLowerBound(i)
                    mean = inputSpace.getLowerBound(i) + 0.5*stdv
                    stdv = stdv/6.0
                    secondStageSampling.append(numpy.random.normal(mean, stdv, numberOfControlActions[i]))
            
            elif samplingMethod == 'LHS_NDS':
                stdv = zeros((inputSpace.globalID+1))
                mean = zeros((inputSpace.globalID+1))
                for i in range(0,inputSpace.globalID+1):
                    stdv[i] = inputSpace.getUpperBound(i) - inputSpace.getLowerBound(i)
                    mean[i] = inputSpace.getLowerBound(i) + 0.5*stdv[i]
                    stdv[i] = stdv[i]/6.0
                    currentSamples = lhs(1, numberOfControlActions[i], 'center')
                    secondStageSampling.append(norm(loc=mean[i], scale=stdv[i]).ppf(currentSamples))
            
            elif samplingMethod == 'LHS_gamma':
                # If ForwarDist is True: the maximum of the gamma distribution will lay near the lower bound
                # If ForwarDist is False: the maximum of the gamma distribution will lay near the upper bound
                forwardDist = True
                # This parameter changes the skewness of the distribution
                a = 2.0

                deltaX = gamma.ppf(0.999, a, loc=0, scale=1)-gamma.ppf(0.001, a, loc=0, scale=1)
                loc = zeros((inputSpace.globalID+1))
                scale = zeros((inputSpace.globalID+1))
                for i in range(0,inputSpace.globalID+1):
                    loc[i] = inputSpace.getLowerBound(i)
                    scale[i] = (inputSpace.getUpperBound(i)-inputSpace.getLowerBound(i))/deltaX                   
                    currentSamples = lhs(1, numberOfControlActions[i], 'center')
                    currentSamples = gamma.ppf(currentSamples, a, loc[i], scale[i])
                    if forwardDist:
                        secondStageSampling.append(currentSamples)
                    else:
                        secondStageSampling.append(inputSpace.getUpperBound(i) - (currentSamples - inputSpace.getLowerBound(i)))

            self.controlSamples = zeros((inputSpace.globalID+1,numberOfSamples+1))
            for i in range(inputSpace.globalID+1):
                controlAction = 0
                for j in range(numberOfSamples+1):
                    if firstStageSampling[i,j] == 1:
                        self.controlSamples[i,j] = secondStageSampling[i][controlAction]
                        controlAction += 1
                    elif firstStageSampling[i,j] == 0:
                        self.controlSamples[i,j] = self.controlSamples[i,j-1]
            
            self.controlSamplesFullTimeSpace = zeros((inputSpace.globalID+1,len(self.time['values'])))
            index = [0]
            for j in range(1,len(self.sampleTime['values'])):
                index.append(numpy.where(self.time['values']==self.sampleTime['values'][j])[0][0])
                for i in range(inputSpace.globalID+1):
                    self.controlSamplesFullTimeSpace[i,index[j-1]:index[j]+1] = self.controlSamples[i,j-1]


        else:
            print "ERROR: Select a valid control sampling method!"
            exit()

    def plotSamples(self, sample=0, plotHistogram=False):
    
        plt.plot(self.time['values'],self.controlSamplesFullTimeSpace[sample,:])
        plt.show()

        if plotHistogram:
            plt.hist(numpy.unique(self.controlSamples[sample,:]))
            plt.show()
    
    def getSamples(self):
        sampleResults = {'controlSamples': self.controlSamples, 'sampleTime': self.sampleTime, 'controlSamplesFullTime': self.controlSamplesFullTimeSpace}
        return sampleResults, self.time
        

        


        
        
        



        
        