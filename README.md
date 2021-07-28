----> Deep Learning based Feedback <---

Description:
The script is intended to be used analyzing a video of a English learner practicing English speaking.
This script extract visual and voice features.

MADE and TESTED for  OS->UBUNTU 16.04 or higher.

Features:
	NLP:
		1. transcription
		2. duration of the video
		3. word start timestamp
		4. word end timestamp
		5. pause between words

	CV: 
		1. blinks per minute
		2. eye contact not made per minute
		3. brightness/ illumination
		
Additional features:
	1. Stop execution if volume below threshold. (Average volume for whole video.)
	

1.	>>	Dependencies and Installation

This script is meant to work for Python3.7. So, it is included.

STEP 1:  Change directory to script folder. Open terminal (INSTALLATION MUST BE DONE FROM SCRIPT DIRECTORY)
	1.1 --> $ cd /path/to/the/ai-feedback
			$ whereis sh

			If the above statement returns "/bin/sh" , which is the case for most OS, then proceed to STEP 2.
	1.2 --> If where sh is not as mentioned in 1.1, copy path of shown of sh.
	1.3 --> Open setup.sh file with and editor, and in the 1st line paste the above copied path after #!
			Example: If your whereis sh shows "/something/sh", then 1st line of setup.sh should be: #!/something/sh

STEP 2:  Install python3.7 by using the commands:

		$ sudo apt-get install python3.7
		$ sudo apt-get install python3-pip
		$ sudo python3.7 -m pip install pip
		$ sudo apt-get install python3.7-dev
		$ python3.7 -m pip install --upgrade pip

STEP 3:  Now, using terminal, type following command:

			   $ sudo bash setup.sh

		This will install major dependencies of the pip dependencies.
	  
STEP 4:  Using terminal, use the following command:

				$ python3.7 -m pip install -r requirements.txt

		This will install all pip module dependencies.
		
After these steps, the script is ready for execution.

2.	>> USAGE
## USE feedback.py for .mp4 videos & feed.py for .webm videos.

###Excecuting when current directory is directory of the script::
Video can be stored anywhere, but must be provided in absolute path:: /path/to/the/file/video.mp4 or
/path/to/the/file/video.webm, thereafter just call the script:

$ python3.7 feeback.py video.mp4

$ python.37 feed.py video.webm

###Executing when current directory is DIFFERENT from the script:

$ python3.7 /absolute/path/to/the/script/directory/feedback.py /absolute/path/to/the/file/video.mp4

$ python3.7 /absolute/path/to/the/script/directory/feed.py /absolute/path/to/the/file/video.webm


You can simply call this script from your programming language too.
For example,
import os
command= "python3.7 feedback.py " + 
os.system()




3.	>> OUTPUT    (STORED AT < DIRECTORY-OF-SCRIPT/json/* >)

The output generated contains parameters from inference. They are stored in json directory within the directory containing this script.
The format of json is explained with example below:

EXAMPLE:

{
    "nlp": {
        "transcription": "ALLO GOOD AFTERNOON MY NO MISSIER AND I ENTER AND AND I AM FROM MITO",
        "duration": "9.01",
        "list": {
            "ALLO": "0.4206,0.5808,0",
            "GOOD": "0.9814,1.1216,0.4006",
            "AFTERNOON": "1.2218,1.6624,0.1001",
            "MY": "2.2633,2.3434,0.6009",
            "NO": "2.4836,2.5237,0.1402",
            "MISSIER": "2.6238,3.6453,0.1001",
            "AND": "7.3506,7.5309,0.9414",
            "I": "7.7112,7.7112,0.1803",
            "ENTER": "5.3077,5.7283,0.4607",
            "AM": "8.0516,8.1518,0.3405",
            "FROM": "8.292,8.4723,0.1402",
            "MITO": "8.6725,9.013,0.2003"
        }
    },
    "cv": {
        "blink-counter": "159.77",
        "non-eye-contact-counter": "53.26",
        "brightness": "LOW"
    }
}

Term Descriptions:
1.	nlp	=>  It stand for Natural Language Processing, the NLP part of the analysis.
	1.1.	transcription =>  It is the words that were understood by the NLP model from give source (video).
	1.2.	duration =>  It is the duration of the video in SECONDS.
	1.3.	list =>  This is a list of timestamps and pauses in for each SPOKEN WORD recognized by the model as in transcription. 
			The sequence in the CSV string is as follow:  START_TIMESTAMP, END_TIMESTAMP, PAUSE_BETWEEN_CURRENT_AND_LAST_WORD
2.	cv =>  It stands for Computer Vision, the visual processing part of the inference.
	2.1. blink-counter =>  Number of blinks PER MINUTE
	2.2. non-eye-contact-counter =>  Time when not making eye contact to the duration of video in PERCENTAGE.
	2.3. brightness =>  Brightness of the video. It considers average brightness across frames. It has three possible values: LOW, FINE, HIGH. 
				
				


		
Update Log:
Date: 2021/07/19
1. Added support for web with feed.py
2. Updated threshold values for brightness and loudness.				
				
				
				
				
				
---------------------->> Additional Resources
1. To run this script from anywhere in the Ubuntu system, refer to this link: https://www.geeksforgeeks.org/run-python-script-from-anywhere-in-linux/
				
				