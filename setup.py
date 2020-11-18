from setuptools import setup, find_packages
import itsfm

 # link to test upload and fresh install on Test PyPi https://packaging.python.org/guides/using-testpypi/
 
version_number = itsfm.__version__

setup(name='itsfm',
     version=version_number,
     description='Identify, Track and Segment sounds by Frequency and its Modulation',
     long_description=open('README.md').read(),
     long_description_content_type="text/markdown",
     url='https://github.com/thejasvibr/itsfm.git',
     author='Thejasvi Beleyur',
     author_email='thejasvib@gmail.com',
     license='MIT',
     packages=find_packages(),
     install_requires=['numpy>1.15','pandas','soundfile',
        'scipy','matplotlib', 'scikit-image', 'tftb','tqdm'],
     zip_safe=False,
	 include_package_data=True,
     classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio',
		'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3'
		])
