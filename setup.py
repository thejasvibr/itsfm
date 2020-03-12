from setuptools import setup, find_packages

setup(name='measure_horseshoe_bat_calls',
     version='1.0.0',
     description='Measure single Cf bat calls, or sounds like it',
     long_description=open('README.md').read(),
     long_description_content_type="text/markdown",
     url='https://github.com/thejasvibr/measure_horseshoe_bat_calls.git',
     author='Thejasvi Beleyur',
     author_email='thejasvib@gmail.com',
     license='MIT',
     packages=find_packages(),
     install_requires=['numpy','pandas','soundfile','peakutils',
        'scipy','matplotlib', 'PyWavelets '],
     zip_safe=False,
	 include_package_data=True,
	 package_data={'':['data/*.WAV']},
     classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio'])
