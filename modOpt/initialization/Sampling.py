'''
Created on Jan 21, 2012

@author: j.weigert
'''

import numpy
import scipy.stats
#import matplotlib.pyplot as plt
import sys
from decimal import Decimal
import time as timefunc

#sampling algorithms
from modOpt.initialization.SamplingMethods.hammersley import hammersley_sequence
from modOpt.initialization.SamplingMethods.sobol import i4_sobol_generate
from modOpt.initialization.SamplingMethods.latin_hypercube import lhs

sampling_methods_list = ['hammersley', 'latin_hypercube', 'sobol']
sequential_methods_list = ['hammersley', 'sobol']

class Variable_Sampling:
    '''
    method creates a multidimensional distribution based on the specified method in sampling object

    Parameter
    ---------

    sampling_object: VarListType object
        variable list containing sampling information and variables for which samples are generated
    number_of_samples: int
        number of generated samples

    '''

    def __init__(self, sampling_object, number_of_samples=10, samples_to_skip=0):

        
        if hasattr(sampling_object, 'adaptiveSampling'):
            if sampling_object.adaptiveSampling and not sampling_object.samplingMethod in sequential_methods_list:
                print('\nERROR: Select a valid sampling method! Must be sequential, hence either Hammersley or Sobol \n')
                sys.exit()        
        self.sampling_object = sampling_object
        self.seed = sampling_object.seed
        self.samples = []
        self.number_of_samples = number_of_samples
        self.samples_to_skip = samples_to_skip
        self.samples_to_skip_seed = sampling_object.numberOfSamples #this is where it comes in hand that it is not maxNumberOfSamples for Adaptive type

    def create_samples(self):
        '''
        method creates samples based on passed variable list object

        Returns
        -------

        out: list
            list of distribution samples

        '''
        print('\nVariable Sampling Started')

        if not self.sampling_object.performSampling:
            self.samples = [[]]
            for var in range(self.sampling_object.globalID+1):
                self.samples[-1].append(self.sampling_object.varlist[var].value)
            return self.samples

        if self.sampling_object.samplingMethod in sequential_methods_list:
            samples_uniform = getattr(self, '_create_' + self.sampling_object.samplingMethod + '_samples')(self.number_of_samples, self.samples_to_skip)
        else:
            samples_uniform = getattr(self, '_create_' + self.sampling_object.samplingMethod + '_samples')(self.number_of_samples)

        self.samples = numpy.transpose(self._distribution_transformation(samples_uniform)).tolist()
        
        return self.samples
    
    def create_samples_for_adaptive_sampling(self, number_of_samples, samples_to_skip):

        samples_uniform = getattr(self, '_create_' + self.sampling_object.samplingMethod + '_samples')(number_of_samples, samples_to_skip)
        samples = numpy.transpose(self._distribution_transformation(samples_uniform)).tolist()
        return samples

    def _create_hammersley_samples(self, number_of_samples, samples_to_skip):
        if self.seed == None: self.seed=0
        samples = hammersley_sequence(1 + samples_to_skip + (self.seed * self.samples_to_skip_seed), samples_to_skip + number_of_samples + (self.seed * self.samples_to_skip_seed), self.sampling_object.globalID+2, 1)
        return samples[1:]
    
    def _create_sobol_samples(self, number_of_samples, samples_to_skip):
        if self.seed == None: self.seed=0
        samples = i4_sobol_generate(self.sampling_object.globalID+1, number_of_samples, 2 + samples_to_skip + (self.seed * self.samples_to_skip_seed))
        return samples
    
    def _create_latin_hypercube_samples(self, number_of_samples):

        samples = lhs(n=self.sampling_object.globalID+1, samples=number_of_samples, criterion='center', seed=self.seed)
        return numpy.transpose(samples)
    
    def _distribution_transformation(self, samples_uniform):
        # transforms the uniform distribution in samples_uniform to a specified probability distribution from scipy.stats
        samples_distribution = []
        for var in range(self.sampling_object.globalID+1):
            if self.sampling_object.varlist[var].samplingDistribution:
                distribution_args = self.sampling_object.varlist[var].distributionParams+(self.sampling_object.varlist[var].loc, self.sampling_object.varlist[var].scale)
                samples_distribution.append(getattr(scipy.stats, self.sampling_object.varlist[var].samplingDistribution)(*distribution_args).ppf(samples_uniform[var]))
            else:
                distribution_args = self.sampling_object.distributionParams+(self.sampling_object.varlist[var].loc, self.sampling_object.varlist[var].scale)
                samples_distribution.append(getattr(scipy.stats, self.sampling_object.samplingDistribution)(*distribution_args).ppf(samples_uniform[var]))
        
        return samples_distribution
    
    # def plot_samples(self, dim1=None, dim2=None, plot_histogram=False):
    #     '''
    #     method plots generated samples 2D

    #     Parameter
    #     ---------

    #     dim1, dim2: int, optional
    #         if specified the the two dimensions of the sample space are plotted 2D
    #         if None all combinations are plotted
    #     plot_histogram: bool, optional
    #         if True histogram of all dimensions is plotted
    #     '''

    #     if isinstance(dim1, int) and isinstance(dim2, int):
    #         plt.plot([x[dim1] for x in self.samples], [x[dim2] for x in self.samples],'*')
    #         plt.title('Sample {} over Sample {}'.format(dim1,dim2))
    #         plt.show()
    #     else:
    #         for dim1 in range(len(self.samples[0])):
    #             for dim2 in range(dim1, len(self.samples[0])):
    #                 if dim1 != dim2:
    #                     plt.figure()
    #                     plt.plot([x[dim1] for x in self.samples], [x[dim2] for x in self.samples],'*')
    #                     plt.title('Sample {} over Sample {}'.format(dim1,dim2))
    #         plt.show()
        
    #     if plot_histogram:
    #         for sample in range(len(self.samples[0])):
    #             plt.figure()
    #             plt.hist([x[sample] for x in self.samples])
    #             plt.title('Histogram of Sample {}'.format(sample))
#            plt.show()

class Control_Sampling():
    '''
    method creates amplitude pseudo-random binary sequence (APRBS) samples for specified controls

    Parameter
    ---------

    sampling_object: VarTypeList object
        control variable list containing sampling information and variables for which APRBS samples are generated
    time: list
        list of time points at which the snapshot generation is supposed to generate samples

    '''

    def __init__(self, sampling_object, sampling_frequencies):

        self.sampling_object = sampling_object
        # self.sampling_frequencies = sampling_frequencies
        self.sampling_object.sampling_frequencies = sampling_frequencies

        self._time_unit_check()

        self._check_control_sampling_frequencies()

        self.create_possible_sample_time()

        self.sampling_object.numberOfSamples = len(self.possible_sample_time) - 1

    def create_possible_sample_time(self):
        '''
        method creates possible sample times based on the sampling frequencies dict
        '''

        self.possible_sample_time  = [0.0]
        for freq in self.sampling_object.sampling_frequencies['values']:
            for _  in range(freq[1]):
                self.possible_sample_time.append(round(self.possible_sample_time[-1] + freq[0], 9))
    
    def create_samples(self):
        '''
        method creates APRBS samples based on passed control object

        Returns
        -------

        out: dict, list
            dictionary of APRBS samples, list with all snapshot generation times

        '''

        print("\nControl sampling started")

        if not self.sampling_object.performSampling:
            self._no_sampling()
        else:
            self._perform_sampling()
        
        self.sampling_object.sampleData =  {'controlSamples': self.control_samples, 'sampleTime': self.sample_time}

    def _perform_sampling(self):
        
        # # list of time points where control changes are possible including time point 0
        # possible_sample_time = numpy.linspace(self.time['values'][0], self.time['values'][-1], self.sampling_object.numberOfSamples+1).tolist()[:-1]

        # sampling of binary part of APRBS
        binary_sampling = []
        for var in range(self.sampling_object.globalID+1):
            numpy.random.seed(self.sampling_object.varlist[var].binarySamplingSeed)
            binary_sampling.append(numpy.random.randint(2, size=self.sampling_object.numberOfSamples).tolist())

        # list of time points where controls change after binary sampling for each control separate
        self.separate_sample_time = []
        for var in range(self.sampling_object.globalID+1):
            self.separate_sample_time.append([self.possible_sample_time[0]])
            for t in range(len(binary_sampling[var])):
                if binary_sampling[var][t]:
                    self.separate_sample_time[-1].append(self.possible_sample_time[t+1])
        
        # list of control values at the regarding time points after amplitude sampling
        self.separate_control_values = []
        for var in range(self.sampling_object.globalID+1):
            amplitude_sampling = Variable_Sampling(self.sampling_object, sum(binary_sampling[var])+1)
            self.separate_control_values.append([x[var] for x in amplitude_sampling.create_samples()])

        # dictionary with combined time points where controls change
        self.sample_time = []
        for var in range(self.sampling_object.globalID+1):
            self.sample_time = self.sample_time + self.separate_sample_time[var]
        # self.sample_time = {'values': numpy.around(numpy.unique(self.sample_time), decimals=9).tolist(), 'engUnit': self.sampling_object.sampling_frequencies['engUnit']}    
        self.sample_time = {'values': numpy.unique(self.sample_time).tolist(), 'engUnit': self.sampling_object.sampling_frequencies['engUnit']}    
        
        # # list of time points with control changes combined with predefined time list
        # self.full_time = {'values': numpy.unique(self.sample_time['values']+self.time['values']).tolist(), 'engUnit': self.time['engUnit']}
    

        # generation of control list at sample time
        self.control_samples = Control_Sampling.control_list_generation(self.separate_control_values, self.separate_sample_time, self.sample_time['values'])

        # self.control_samples_full_time = self._control_list_generation(self.separate_control_values, self.separate_sample_time, self.full_time['values'])
    
    def _no_sampling(self):
        
        # generation of necesarry time dictionaries
        self.sample_time = []
        for var in range(self.sampling_object.globalID+1):
            self.sample_time = self.sample_time + self.sampling_object.varlist[var].time['values']
        self.sample_time = {'values': numpy.unique(self.sample_time).tolist(), 'engUnit': self.sampling_object.varlist[0].time['engUnit']}
        # self.full_time = {'values': numpy.unique(self.sample_time['values']+self.time['values']).tolist(), 'engUnit': self.time['engUnit']}

        # generation of control list at sample time
        self.control_samples = Control_Sampling.control_list_generation([x.value for x in self.sampling_object.varlist], [x.time['values'] for x in self.sampling_object.varlist], self.sample_time['values'])

        # # generation of control list at full time
        # self.control_samples_full_time = self._control_list_generation([x.value for x in self.sampling_object.varlist], [x.time['values'] for x in self.sampling_object.varlist], self.full_time['values'])
    
    @staticmethod
    def control_list_generation(control_changes, change_times, time_list):
        # generates a list with controls at the time points of time_list 
        # based on the controls in control_changes at the times change_times

        control_samples = []
        for var in range(len(control_changes)):
            counter = -1
            control_samples.append([])
            for t in range(len(time_list)):
                if time_list[t] in change_times[var]:
                    counter = counter + 1
                control_samples[-1].append(control_changes[var][counter])
        
        return control_samples
        
    def _time_unit_check(self):
        # checks if the time unit in all time dicts is equal
        for var in range(self.sampling_object.globalID+1):
            if self.sampling_object.sampling_frequencies['engUnit'] != self.sampling_object.varlist[var].time['engUnit']:
                print('\nERROR: time unit of control {} is not equal to unit of overall time unit\n'.format(self.sampling_object.varlist[var].varName))
                exit()
            # checks if there is a time point 0.0 if no sampling is performed
            if not self.sampling_object.performSampling:
                if self.sampling_object.varlist[var].time['values'][0] > 0.0:
                    print('\nERROR: No control value at time 0.0 in {}\n'.format(self.sampling_object.varlist[var].varName))
                    exit()
    
    def _check_control_sampling_frequencies(self):
        # check whether higher sampling frequencies are whole multiples of lowest frequency
        # check whether sampling frequencies are sorted from high to low
        f = self.sampling_object.sampling_frequencies['values']

        if len(f) == 1:
            return
        
        if not all(f[i][0] <= f[i+1][0] for i in range(len(f)-1)):
            sys.exit('ERROR: control frequencies are not sorted (from high to low)')
        
        if not all(Decimal(str(f[i+1][0]))%Decimal(str(f[0][0])) == 0.0 for i in range(len(f)-1)):
            sys.exit('ERROR: Not all control frequencies are whole multiples of the lowest frequency')

    
    def plot_samples(self, control=None, plot_histogram=False):
        '''
        method plots APRBS generated APRBS samples over time

        Parameter
        ---------

        control: int, optional
            if specified only this control is plotted, otherwise all controls are plotted
        plot_histogram: bool, optional
            if True histogram of plotted controls is plotted
        '''

        # if isinstance(control, int):
        #     plt.plot(self.sample_time['values'], self.control_samples[control])
        #     plt.title('Trajectory of control {}, {}'.format(control, self.sampling_object.varlist[control].varName))
        #     plt.show()
        # else:
        #     for var in range(self.sampling_object.globalID+1):
        #         plt.figure()
        #         plt.plot(self.sample_time['values'], self.control_samples[var])
        #         plt.title('Trajectory of control {}, {}'.format(var, self.sampling_object.varlist[var].varName))
        #     plt.show()
        
        # if plot_histogram:
        #     if isinstance(control, int):                                                          
        #         plt.hist(self.control_samples[control])
        #         plt.title('Histogram of control {}, {}'.format(control, self.sampling_object.varlist[control].varName))
        #         plt.show()
        #     else:
        #         for var in range(self.sampling_object.globalID+1):
        #             plt.figure()
        #             plt.hist(self.control_samples[var])
        #             plt.title('Histogram of control {}, {}'.format(var, self.sampling_object.varlist[var].varName))
        #         plt.show()

class Control_Sampling_Simulink():
    '''
    method creates amplitude pseudo-random binary sequence (APRBS) samples for specified controls
    specifically for Sinulink simulations, where parallel computing is executed inside the model wrapper

    Parameter
    ---------

    sampling_object: VarTypeList object
        control variable list containing sampling information and variables for which APRBS samples are generated
    time: list
        list of time points at which the snapshot generation is supposed to generate samples

    '''

    def __init__(self, sampling_object, time={}, multifreqSampling=False):

        self.sampling_object = sampling_object
        self.time = time
        self.multifreqSampling = multifreqSampling

        self._time_unit_check()

    def create_samples(self):
        '''
        method creates APRBS samples based on passed control object

        Returns
        -------

        out: dict, list
            dictionary of APRBS samples, list with all snapshot generation times

        '''

        print("\nControl sampling started")

        if not self.sampling_object.performSampling:
            self._no_sampling()
        else:
            self._perform_sampling()
        
        return {'controlSamples': self.control_samples, 'sampleTime': self.sample_time, 'controlSamplesFullTime': self.control_samples_full_time}, self.full_time

    def _perform_sampling(self):
        
        # list of time points where control changes are possible including time point 0
        if not self.multifreqSampling:
            possible_sample_time = numpy.around(numpy.linspace(self.time['values'][0], self.time['values'][-1], self.sampling_object.numberOfSamples+1),6).tolist()[:-1]
        else:
            possible_sample_time = self.create_multifreq_sampletime(self.multifreqSampling)
        # sampling of binary part of APRBS
        binary_sampling = []
        for var in range(self.sampling_object.globalID+1):
            numpy.random.seed(self.sampling_object.varlist[var].binarySamplingSeed)
            binary_sampling.append(numpy.random.randint(2, size=len(possible_sample_time)-1).tolist())
        # list of time points where controls change after binary sampling for each control separate
        self.separate_sample_time = []
        for var in range(self.sampling_object.globalID+1):
            self.separate_sample_time.append([possible_sample_time[0]])
            for t in range(len(binary_sampling[var])):
                if binary_sampling[var][t]:
                    self.separate_sample_time[-1].append(possible_sample_time[t+1])
        
        # list of control values at the regarding time points after amplitude sampling
        self.separate_control_values = []
        for var in range(self.sampling_object.globalID+1):
            amplitude_sampling = Variable_Sampling(self.sampling_object, sum(binary_sampling[var])+1)
            self.separate_control_values.append([x[var] for x in amplitude_sampling.create_samples()])

        # dictionary with combined time points where controls change
        self.sample_time = []
        for var in range(self.sampling_object.globalID+1):
            self.sample_time = self.sample_time + self.separate_sample_time[var]
        self.sample_time = {'values': numpy.unique(self.sample_time).tolist(), 'engUnit': self.time['engUnit']}    
        
        # list of time points with control changes combined with predefined time list
        self.full_time = {'values': numpy.unique(self.sample_time['values']+self.time['values']).tolist(), 'engUnit': self.time['engUnit']}
        
        # check whether full_time is equidistant
        # self._check_equidistant_time(self.full_time) @obsolete: this is no longer necessary, we will fix this afterwards, the simulation should decide on the reporting intervals

        # generation of control list at sample time
        self.control_samples = self._control_list_generation(self.separate_control_values, self.separate_sample_time, self.sample_time['values'])
        tic1=timefunc.time()
        print("generation of controllist at fulltime sampling start")
        self.control_samples_full_time = self.control_samples
        #self.control_samples_full_time = self._control_list_generation(self.separate_control_values, self.separate_sample_time, self.full_time['values'])
        print("generation of controllist at fulltime sampling",timefunc.time()-tic1)
    
    def _no_sampling(self):
        
        # generation of necesarry time dictionaries
        self.sample_time = []
        for var in range(self.sampling_object.globalID+1):
            self.sample_time = self.sample_time + self.sampling_object.varlist[var].time['values']
        self.sample_time = {'values': numpy.unique(self.sample_time).tolist(), 'engUnit': self.time['engUnit']}
        self.full_time = {'values': numpy.unique(self.sample_time['values']+self.time['values']).tolist(), 'engUnit': self.time['engUnit']}

        # check whether full_time is equidistant
        self._check_equidistant_time(self.full_time)

        # generation of control list at sample time
        self.control_samples = self._control_list_generation([x.value for x in self.sampling_object.varlist], [x.time['values'] for x in self.sampling_object.varlist], self.sample_time['values'])

        # generation of control list at full time
        tic2=timefunc.time()
        self.control_samples_full_time = self.control_samples
        #self.control_samples_full_time = self._control_list_generation([x.value for x in self.sampling_object.varlist], [x.time['values'] for x in self.sampling_object.varlist], self.full_time['values'])
        print("generation of controllist at fulltime_nosampling",timefunc.time()-tic2)

    def _control_list_generation(self, control_changes, change_times, time_list):
        # generates a list with controls at the time points of time_list 
        # based on the controls in control_changes at the times change_times
        tic3 = timefunc.time()
        print("tic3")
        control_samples = []
        for var in range(self.sampling_object.globalID+1):
            counter = -1
            control_samples.append([])
            for t in range(len(time_list)):
                if time_list[t] in change_times[var]:
                    counter = counter + 1
                control_samples[-1].append(control_changes[var][counter])
        print("controllistgeneration",timefunc.time()-tic3)
        return control_samples

    def _time_unit_check(self):
        # checks if the time unit in all time dicts is equal
        for var in range(self.sampling_object.globalID+1):
            if self.time['engUnit'] != self.sampling_object.varlist[var].time['engUnit']:
                print('\nERROR: time unit of control {} is not equal to unit of overall time unit\n'.format(self.sampling_object.varlist[var].varName))
                exit()
            # checks if there is a time point 0.0 if no sampling is performed
            if not self.sampling_object.performSampling:
                if self.sampling_object.varlist[var].time['values'][0] > 0.0:
                    print('\nERROR: No control value at time 0.0 in {}\n'.format(self.sampling_object.varlist[var].varName))
                    exit()
    
    def _check_equidistant_time(self, x):
        # check whether time entries are equidistant --> important for blackbox modeling
        delta_t = x['values'][1] - x['values'][0]
        tic4=timefunc.time()
        for i in range(2, len(x['values'])):
            if abs((x['values'][i] - x['values'][i-1]) - delta_t) >= 2e-6:
                answer = input(
                    '\nWARNING: The combined time vector of sample and control times is not equidistant!\nThis can cause problems in system identification.\nAre you sure you want to continue? Please Enter (y/n)')
                if answer == 'n' or answer == 'N':
                    print('\nSNAPSHOT GENERATION TERMINATED')
                    exit()
                if answer == 'y' or answer == 'Y':
                    print('\nSNAPSHOT GENERATION CONTINUES')
                    break
        print("check equidistant time",timefunc.time()-tic4)

    def create_multifreq_sampletime(self,multiSamples):
        #multiSamples = [(numofSamples1,tc1),(numofSamples2,tc2),...]
        possible_sample_time = [self.time['values'][0]]
        tic5=timefunc.time()
        for tup in multiSamples:
            for num in range(tup[0]):
                possible_sample_time.append(numpy.around(possible_sample_time[-1]+tup[1],6))
        print("multisample",timefunc.time()-tic5)
        return possible_sample_time

    def plot_samples(self, control=None, plot_histogram=False):
        '''
        method plots APRBS generated APRBS samples over time

        Parameter
        ---------

        control: int, optional
            if specified only this control is plotted, otherwise all controls are plotted
        plot_histogram: bool, optional
            if True histogram of plotted controls is plotted
        '''

        # if isinstance(control, int):
        #     plt.step(self.sample_time['values'], self.control_samples[control],where='post')
        #     #plt.plot(self.sample_time['values'], self.control_samples[control])
        #     plt.title('Trajectory of control {}, {}'.format(control, self.sampling_object.varlist[control].varName))
        #     plt.show()
        # else:
        #     for var in range(self.sampling_object.globalID+1):
        #         plt.figure()
        #         plt.step(self.sample_time['values'], self.control_samples[var],where='post')
        #         #plt.plot(self.sample_time['values'], self.control_samples[var])
        #         plt.title('Trajectory of control {}, {}'.format(var, self.sampling_object.varlist[var].varName))
        #     plt.show()
        
        # if plot_histogram:
        #     if isinstance(control, int):                                                          
        #         plt.hist(self.control_samples[control])
        #         plt.title('Histogram of control {}, {}'.format(control, self.sampling_object.varlist[control].varName))
        #         plt.show()
        #     else:
        #         for var in range(self.sampling_object.globalID+1):
        #             plt.figure()
        #             plt.hist(self.control_samples[var])
        #             plt.title('Histogram of control {}, {}'.format(var, self.sampling_object.varlist[var].varName))
        #         plt.show()
