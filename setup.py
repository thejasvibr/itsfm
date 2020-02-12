from setuptools import setup

setup(name='measure_CF_calls',
      version='0.0.0',
      description='Measure single Cf bat calls, or sounds like it',
      url='https://github.com/thejasvibr/measure_horseshoe_bat_calls.git',
      author='Thejasvi Beleyur',
      author_email='thejasvib@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
     install_requires=['numpy','pandas','soundfile','peakutils',
        'scipy','matplotlib'],
      zip_safe=False)
