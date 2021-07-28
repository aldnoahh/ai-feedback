import sys
import os
import textstat
import cv2
from gaze_tracking import GazeTracking
import numpy as np
from numpy.linalg import norm
import os
import sys
from itertools import groupby
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pydub import AudioSegment
import numpy as np
import soundfile as sfile
import math
from time import time
import multiprocessing
from readability import Readability
import json
from time import time
import pyloudnorm as pyln
import io
from google.cloud import speech
from google.cloud import storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ubuntu/ai-feedback/keys/taplinguav20-b812c6ad28de.json'
start= time()


manager = multiprocessing.Manager()
cv_out = manager.list()
nlp_out = manager.list()
stt_out= manager.list()

src1= sys.argv[1]
if "mp4" in src1:
    srcc= src1.replace('.mp4','.webm')
    webb= "ffmpeg -y -i "+src1+" -c:v libvpx-vp9 -crf 4 -b:v 0 "+srcc
    os.system(webb)
    src= srcc
else:
    src= src1




def cv(cv_out):
    def bright(img):
        if len(img.shape) == 3:
            # Colored RGB or BGR (*Do Not* use HSV images with this function)
            # create brightness with euclidean norm
            return np.average(norm(img, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            return np.average(img)
            


    #src = sys.argv[1]
    gaze = GazeTracking()
    #webcam = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(src)
    blink_counter=0
    notlookingstraight_counter=0
    brightness=[0,0,0]        #high, mid, low
    counter=0
    while  (cap.isOpened()):
        # We get a new frame from the webcam
        ret, frame = cap.read()
        if ret == True:
            fr=frame
            counter+=1

            # We send this frame to GazeTracking to analyze it
            gaze.refresh(frame)

            frame = gaze.annotated_frame()

            if gaze.is_blinking():
                blink_counter+=1
            if gaze.is_center() == False:
                notlookingstraight_counter+=1

            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            #cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            #cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

            #cv2.imshow("Demo", frame)

            #if cv2.waitKey(1) == 27:
            #    break
            
          
            #print(bright(fr))
            if bright(fr) > 186:  
                brightness[0]+=1   
            elif bright(fr) > 50:
                brightness[1]+=1
            else:
                brightness[2]+=1

        else:
            break
       
    cap.release()




    br=""
    #print("----------------------->>>>  {}".format(brightness))
    id = brightness.index(max(brightness))
    if id==0:
        br="HIGH"
    if id==1:
        br="FINE"
    if id==2:
        br="LOW"
    """
    fname = os.path.basename(src).replace('.mp4','.txt')
    with open(fname, 'w') as f:
        print("Number-of-times-blinking,Number-of-times-Not-looking-straight,average-illumination",file=f)
        print("{},{},{}".format(blink_counter,notlookingstraight_counter,br),file=f)
    """
    #cv= [blink_counter, notlookingstraight_counter, br]
    #return cv
    cv_out.append(blink_counter)
    cv_out.append(notlookingstraight_counter)
    cv_out.append(br)
    cv_out.append(counter)


def nlp(nlp_out):
    # files                                                                         
    #src = sys.argv[1]
    
    nam= os.path.basename(src).replace('.webm','.wav')
    dst=  os.path.dirname(os.path.realpath(__file__))+ "/wav/"+nam
    # convert wav                                                           
    sound =  AudioSegment.from_file(src)
    #sound = sound.set_channels(2)
    #sound = sound.set_frame_rate(16000)
    sound.export(dst, format="wav")
    
    duration= librosa.get_duration(filename=dst)
    
    if duration >20:
        fac= int((duration//20)) +1
        t1=0
        t2=20
        filenames=[0]
        for i in range(fac):
            t11 = t1 * 1000 #Works in milliseconds
            t22 = t2 * 1000
            newAudio = AudioSegment.from_wav(dst)
            newAudio = newAudio[t11:t22]
            dstt= os.path.dirname(os.path.realpath(__file__))+ "/wav/"+ str(i+1)+nam
            newAudio.export(dstt, format="wav") #Exports to a wav file in the current path.
            filenames.append(dstt)
            if t22+20000 > int(duration*1000):
                t1+=20
                t2= abs(duration)
            else:
                t1+=20
                t2+=20
    else:
        filenames= [0,dst]
    

    filename = dst
    data, rate = sfile.read(filename) # load audio (with shape (samples, channels))
    meter = pyln.Meter(rate) # create BS.1770 meter
    avr = meter.integrated_loudness(data) # measure loudness
    if avr >33.5:
        print(avr)
        print("----------->>>>>  Volume TOO Low, Unable to Process       <<<<<-------------------")
        jdictt= { 'Error': 'Volume was too low.' }
        jfile_p= os.path.basename(src).replace('.webm','.json')
        jfile=  os.path.dirname(os.path.realpath(__file__))+ "/json/"+ jfile_p
        jjfile=  os.path.dirname(os.path.realpath(__file__))+ "/json/json-debug/"+ jfile_p
        json_object = json.dumps(jdictt, indent = 4)
        with open(jjfile, "w") as outfile:
            outfile.write(json_object)
        with open(jfile, "w") as outfile:
            outfile.write(json_object)
        exit(0)
    ### Volume Ranges:: 

  
  
    
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")    
    transcription= []

    for i in range(1, len(filenames)):
 
        audio , rate = librosa.load(filenames[i], sr = 16000)

        input_values = tokenizer(audio, return_tensors = "pt").input_values
        logits = model(input_values).logits
        prediction = torch.argmax(logits, dim = -1)
        transcribe = tokenizer.batch_decode(prediction)[0]
        transcription.append(transcribe)
        
        
    #print(transcription)
    trans= " ".join(transcription)
    print(trans)
    words= trans.split(" ")
        
        
        
    """
    # this is where the logic starts to get the start and end timestamp for each word
    words = [w for w in transcription.split(' ') if len(w) > 0]
    prediction = prediction[0].tolist()
    duration_sec = input_values.shape[1] / rate


    ids_w_time = [(i / len(prediction) * duration_sec, _id) for i, _id in enumerate(prediction)]
    # remove entries which are just "padding" (i.e. no characers are recognized)
    ids_w_time = [i for i in ids_w_time if i[1] != tokenizer.pad_token_id]
    # now split the ids into groups of ids where each group represents a word
    split_ids_w_time = [list(group) for k, group
                        in groupby(ids_w_time, lambda x: x[1] == tokenizer.word_delimiter_token_id)
                        if not k]

    assert len(split_ids_w_time) == len(words)  # make sure that there are the same number of id-groups as words. Otherwise something is wrong

    word_start_times = []
    word_end_times = []
    for cur_ids_w_time, cur_word in zip(split_ids_w_time, words):
        _times = [_time for _time, _id in cur_ids_w_time]
        word_start_times.append(min(_times))
        word_end_times.append(max(_times))
    """ 
    #print(len(words))
    #print(len(word_start_times))
    #print(len(word_end_times))
    #print(transcription)
    #for i in range(len(words)):
    """
    #    print("{}      {}     {}".format(words[i], word_start_times[i], word_end_times[i]))
        
    fname= nam+ ".txt"
    with open(fname, 'w') as f:
        print("transcription",file=f)
        print(transcription,file=f)
        print("word,start timestamp,end timestamp,pause",file=f)   
        for i in range(0,len(words)):
            if i==0:
                print("{},{},{},{}".format(words[i], word_start_times[i], word_end_times[i], 0),file=f)
            else:
                print("{},{},{},{}".format(words[i], word_start_times[i], word_end_times[i], round(word_start_times[i]-word_end_times[i-1],4)),file=f)
        
    print("Execution Successful! Finished in {} seconds.".format( round(time()-start, 3)))
    
    #nlp= [transcription, words, word_start_times, word_end_times]
    nlp_out.append(transcription)
    nlp_out.append(words)
    nlp_out.append(word_start_times)
    nlp_out.append(word_end_times)
    """
    
    
    #nlp= [transcription, words, word_start_times, word_end_times]
    #print(transcription)
    #print("\n\n",gtranscription)
    
    #nlp_out.append(len(words)- len(gword))
    nlp_out.append(duration)
    nlp_out.append(words)
    
  
  
  
def stt(stt_out):

    nam2= os.path.basename(src).replace('.webm','.flac')
    dst2=  os.path.dirname(os.path.realpath(__file__))+ "/flac/"+nam2
    sauce= "ffmpeg -y -i "+ src +" -acodec flac -bits_per_raw_sample 16 -ac 1 -ar 16000 "+ dst2
    os.system(sauce)
    #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    #with io.open(dst2, "rb") as audio_file:
    #    content = audio_file.read()
    # Instantiates a client
    #client = speech.SpeechClient()

    # The name of the audio file to transcribe
    #gcs_uri = "gs://cloud-samples-data/speech/brooklyn_bridge.raw"
    #audio = speech.RecognitionAudio(uri=gcs_uri)
    #audio = speech.RecognitionAudio(content=content)
    print("Uploading......")
    client = storage.Client.from_service_account_json(json_credentials_path='/home/ubuntu/ai-feedback/keys/taplinguav20-f4cb6cbb7775.json')
    bucket = client.get_bucket('tts-taplingua')
    object_name_in_gcs_bucket = bucket.blob(nam2)
    object_name_in_gcs_bucket.upload_from_filename(dst2)
    client = speech.SpeechClient()
    gcs_uri= "gs://tts-taplingua/"+nam2
    audio = speech.RecognitionAudio(uri=gcs_uri)
    
    print("Getting results........")
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-IN",
    )
    
    
    #credentials = service_account.Credentials.from_service_account_file('/home/ubuntu/ai-feedback/taplinguav20-b812c6ad28de.json')
    # Detects speech in the audio file
    #response = client.recognize(config=config, audio=audio)
    #for result in response.results:
    #print("Transcript: {}".format(result.alternatives[0].transcript))
    #response= client.long_running_recognize(config=config, audio=audio)
    
    operation = client.long_running_recognize(config=config, audio=audio)
    print("\nWaiting for operation to complete...")
    result = operation.result(timeout=3000)
    
    #gword=[]
    #gstart=[]
    #gend=[]
    
    gtranscription=""
    for result in result.results:
        alternative = result.alternatives[0]
        #print("Transcript: {}".format(alternative.transcript))
        #print("Confidence: {}".format(alternative.confidence))
        if gtranscription=="":
            gtranscription= gtranscription+ str(alternative.transcript)
        else:
            gtranscription= gtranscription+ " "+str(alternative.transcript)

        #for word_info in alternative.words:
        #    word = word_info.word
        #    start_time = word_info.start_time
        #    end_time = word_info.end_time
        #    gword.append(word)
        #    gstart.append(float(start_time.total_seconds()))
        #    gend.append(float(end_time.total_seconds()))

        #    print("Word: {}, start_time: {}, end_time: {}".format(word,start_time.total_seconds(),end_time.total_seconds()))

    
    gword= gtranscription.split(" ")
    stt_out.append(gtranscription)
    stt_out.append(gword)








    

##############  Calling  
process1 = multiprocessing.Process(target=cv, args=[cv_out])
process2 = multiprocessing.Process(target=nlp, args=[nlp_out])
process3 = multiprocessing.Process(target=stt, args=[stt_out])
process1.start()
process2.start()
process3.start()
process1.join()
process2.join()
process3.join()

blink_counter, notlookingstraight_counter, br, counter= cv_out
#print(nlp_out)
duration,wwords = nlp_out
transcription, words= stt_out
filler_words= len(wwords)-len(words)

if len(words)<1:
    print("\nError: Video is too short OR No words detected! Please try again!")
    jdictt= { 'Error': 'No words were detected.' }
    jfile_p= os.path.basename(src).replace('.webm','.json')
    jfile=  os.path.dirname(os.path.realpath(__file__))+ "/json/"+ jfile_p
    jjfile=  os.path.dirname(os.path.realpath(__file__))+ "/json/json-debug/"+ jfile_p
    json_object = json.dumps(jdictt, indent = 4)
    with open(jjfile, "w") as outfile:
        outfile.write(json_object)
    with open(jfile, "w") as outfile:
        outfile.write(json_object)
    exit(0)
#duration= word_end_times[len(word_end_times)-1]
fps= counter/duration
#print(fps)
#print(counter)
fscore=textstat.flesch_reading_ease(transcription)
if(len(words))>100:
    r = Readability(transcription)
    print("Flesch-Kincaid------------>>>> ",r.flesch_kincaid())
    print("Flesch-------------------->>>> ",r.flesch())

word_counter= round(len(wwords)/duration*60,3)
nlp_dict= {
            "transcription": transcription,
            "duration": str(round(duration,2)),
            "filler-words": str(filler_words),
            "flesch-score":str(fscore),
            "words-per-minute":str(word_counter)
            
}

cv_dict= {
        "blink-counter":str(round((blink_counter/fps)/duration*60,2)),
        "non-eye-contact-counter": str(round(notlookingstraight_counter/counter*100,2)), 
        "brightness": str(br)
}

root= {
        "nlp": nlp_dict,
        "cv": cv_dict
}



#print(root)
#print(len(cv_out))
#print(nlp_out)
jfile_p= os.path.basename(src).replace('.webm','.json')
jfile=  os.path.dirname(os.path.realpath(__file__))+ "/json/"+ jfile_p
jjfile=  os.path.dirname(os.path.realpath(__file__))+ "/json/json-debug/"+ jfile_p
json_object = json.dumps(root, indent = 4)
with open(jjfile, "w") as outfile:
    outfile.write(json_object) 
  
#writing json  
with open(jfile, "w") as outfile:
    outfile.write(json_object)
  
print(json_object)
print(type(json_object))
print("Execution Successful! Finished in {} seconds.".format( round(time()-start, 3)))
print("Output JSON File saved at: {}".format(jfile))
