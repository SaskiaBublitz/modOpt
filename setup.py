from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='modOpt',
      version='0.1',
      description='restructure NLE in order to find its solution(s)',
      long_description = readme(),
      classifiers = ['Programming Language :: Python :: 2.7'],
      url='https://gitlab.tubit.tu-berlin.de/dbta/modOpt.git',
      author='Saskia Bublitz',
      author_email='saskia.bublitz@tu-berlin.de',
      license='TU Berlin',
	  
      packages=['modOpt',
				'modOpt.constraints',
				'modOpt.initialization',
				'modOpt.decomposition',
				'modOpt.scaling',
				'modOpt.solver'],
	  include_package_data=True,
      install_requires=['numpy',
                        'sympy',
                        'mpmath',
                        'casadi',
						'matplotlib'],
                        #TODO: What to do with copy, itertools, time, multiprocessing
      zip_safe=False)