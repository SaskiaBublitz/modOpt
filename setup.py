from setuptools import setup
import re

def readme():
    with open('README.md') as f:
        return f.read()

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)
project_name = 'modOpt'

setup(name=project_name,
      version=get_property('__version__', project_name),
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
			'pyibex',
                        'scipy',
			'scipy',
			'affapy',
			],
                        #TODO: What to do with copy, itertools, time, multiprocessing
      zip_safe=False)
