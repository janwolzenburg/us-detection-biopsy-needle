# us-detection-biopsy-needle
Biopsy needle detection with ultrasonic imaging and a program run on a Raspberry Pi. <br>Student project at Fachhochschule Südwestfalen in Lüdenscheid, Germany


<ul> <h2>Requirements</h2>
	<li> Hardware
		<ul>
			<li>Raspberry Pi 3</li>
			<li>USB Video Grabber</li>
			<li>Needle holder with conducting loops</li>
		</ul>
	</li>
	<li> Python Modules
		<ul>
			<li>math</li>
			<li>numpy</li>
			<li>scipy</li>
			<li>skimage</li>
			<li>matplotlib</li>
			<li>time</li>
			<li>opencv-python</li>
			<li>astropy</li>
			<li>RPi.GPIO</li>
		</ul>
	</li>
	<h2>File descriptions</h2>
	<h3>parameters.py</h3>
	This file holds the necessary variables for image processing and handling. For each system and transducer the image and needle holder properties have to be refreshed. The other parameters affect mostly the processing of a frame for needle processing.
	<h3>functions.py</h3>
	This file contains all the functions used.
	<h3>realtime_process.py</h3>
	This is the main file which contains initialisation and a loop for coninuous realtime image processing.
	<h3>show_frame.py</h3>
	This file's sole purpose is reading .npy files which are written to by the main script and contain the processed frames with the identified needle and tip. The contents of the .npy files are diplayed.
	<h2>How to use?</h2>
	<ol>
		<li>Setup the Raspberry Pi with the required packages drivers for the video grabber</li>
		<li>Find the specific image and needle holder properties to be used in parameters.py</li>
		<li>Start the scripts realtime_process.py andshow_frames.py in this order</li>
		<li>Review the results and change processing parameters if necessary</li>
	</ol>
	For testing purposes previously recorded images can be processesed. Comment all lines in realtime_process.py which handle the video grabber and use cv2.imread() to read images from the specified location. An example can be seen in the code after the start of the section "Repeatet steps". Also the angle has to be considered for testing. In parameters.py angles[1] is the default angle, when no hardware interaction is present.
