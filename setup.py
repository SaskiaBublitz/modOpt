from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='modOpt',
      version='3.2',
      description='restructure NLE in order to find its solution(s)',
      long_description = readme(),
      classifiers = ['Programming Language :: Python :: 3.7'],
      url='https://git.tubit.tu-berlin.de/dbta/modOpt.git',
      author='Saskia Bublitz',
      author_email='saskia.bublitz@tu-berlin.de',
      license='TU Berlin',
	  
      packages=['modOpt',
				'modOpt.constraints',
				'modOpt.initialization',
				'modOpt.decomposition',
				'modOpt.scaling',
				'modOpt.solver',
				'modOpt.tests'],
	  include_package_data=True,
      install_requires=['numpy',
                        'sympy',
                        'mpmath',
                        'casadi',
			'matplotlib',
			'ipopt',
			'pyibex',
                        'scipy',
			'ipopt',
			'scipy',
			'affapy',
			'ax'],
                        #TODO: What to do with copy, itertools, time, multiprocessing
      zip_safe=False)
