import setuptools

setuptools.setup(
    name="us-detection-biopsy-needle-janwolzenburg",
    version="0.0.1",
    author="Jan Wolzenburg",            # @ Fachhochschule Südwestfalen in Lüdenscheid GER 
    author_email="wolzenburg.jan@fh-swf.de",
    description="Biopsy needle detection with ultrasonic imaging and a program run on a Raspberry Pi.",
    url="https://github.com/janwolzenburg/us-detection-biopsy-needle",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',                    # 1.19.2
	'opencv-python',            # 4.4.0.44
        'astropy.convolution'       # 4.0.3
)
