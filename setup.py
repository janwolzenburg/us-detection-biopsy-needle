from setuptools import setup, find_packages

setup(
    name='needle_detection',
    author='Jan Wolzenburg',
    author_email='wolzenburg.jan@fh-swf.de',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        #'cv2',
        'numpy',
        'scipy',
        'matplotlib',
	'opencv-python'
    ],
   # package_data={'all_sky_cloud_detection': ['resources/hipparcos.fits.gz']},
   # setup_requires=['pytest-runner'],
   # tests_require=['pytest'],
)
