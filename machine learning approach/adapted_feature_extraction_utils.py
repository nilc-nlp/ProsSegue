"""
Code with functions to extract prosodic features from audios, specifically pitch and intensity features,
based on Praat and parselmouth.

The following code was extracted and adapted from file feature_extraction_utils.py from https://github.com/uzaymacar/simple-speech-features, 
which contains the extraction of several more features

The references cited there are: 

    Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics, 71, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001
    http://www.fon.hum.uva.nl/praat/manual/Query.html
    http://www.fon.hum.uva.nl/rob/NKI_TEVA/TEVA/HTML/Analysis.html (Some methods are inspired from this reference)
    Feinberg, D. R. (2019, October 8). Parselmouth Praat Scripts in Python. https://doi.org/10.17605/OSF.IO/6DWR3
    http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
    Sample sound file (sample.wav) acquired from http://www.voiptroubleshooter.com/open_speech/american.html

"""

import math
from parselmouth.praat import call


def get_intensity_attributes(sound, time_step=0., min_time=0., max_time=0., pitch_floor=75.,
                             interpolation_method='Parabolic', return_values=False,
                             replacement_for_nan=0.): 
    """
    Function to get intensity attributes such as minimum intensity, maximum intensity, mean
    intensity, and standard deviation of intensity. 
        -> A partir destes valores calcular diferença entre máxima e mínima intensidade, 
        diferença entre máxima e média, e diferença entre mínima e média

    NOTE: Notice that we don't need a unit parameter for intensity as intensity is consistently
    reported as dB SPL throughout Praat. dB SPL is simply dB relative to the normative auditory
    threshold for a 1000-Hz sine wave, 2 x 10^(-5) Pascal.

    NOTE: The standard interpolation method is 'Parabolic' because of the usual non-linearity
    (logarithm) in the computation of intensity; sinc interpolation would be too stiff and may
    give unexpected results.

    :param (parselmouth.Sound) sound: sound waveform
    :param (float) time_step: the measurement interval (frame duration), in seconds (default: 0.)
           NOTE: The default 0. value corresponds to a time step of 0.75 / pitch floor
    :param (float) min_time: minimum time value considered for time range (t1, t2) (default: 0.)
    :param (float) max_time: maximum time value considered for time range (t1, t2) (default: 0.)
           NOTE: If max_time <= min_time, the entire time domain is considered
    :param pitch_floor: minimum pitch (default: 75.)
    :param (str) interpolation_method: method of sampling new data points with a discrete set of
           known data points, 'None', 'Parabolic', 'Cubic', or 'Sinc' (default: 'Parabolic')
    :param (bool) return_values: whether to return a continuous list of intensity values
           from all frames or not
    :param (float) replacement_for_nan: a float number that will represent frames with NaN values
    :return: (a dictionary of mentioned attributes, a list of intensity values OR None)
    """

    # Create Intensity object
    intensity = call(sound, 'To Intensity', pitch_floor, time_step, 'yes')

    attributes = dict()

    min_intensity = call(intensity, 'Get minimum', min_time, max_time, interpolation_method)
    max_intensity = call(intensity, 'Get maximum', min_time, max_time, interpolation_method)
    mean_intensity = call(intensity, 'Get mean',min_time, max_time)

    attributes['e_range'] = max_intensity - min_intensity
    attributes['e_maxavg_diff'] = max_intensity - mean_intensity
    attributes['e_avgmin_diff'] = mean_intensity - min_intensity

    intensity_values = None

    if return_values:
        intensity_values = [call(intensity, 'Get value in frame', frame_no)
                            for frame_no in range(len(intensity))]
        # Convert NaN values to floats (default: 0)
        intensity_values = [value if not math.isnan(value) else replacement_for_nan
                            for value in intensity_values]

    return attributes,  intensity_values

def get_utterance_avg_pitch(sound, pitch_type='preferred', time_step=0.02, min_time=0., max_time=0.,
                         pitch_floor=75., pitch_ceiling=600., unit='Hertz'):

       # Create pitch object
       if pitch_type == 'preferred':
              pitch = call(sound, 'To Pitch', time_step, pitch_floor, pitch_ceiling)
       elif pitch_type == 'cc':
              pitch = call(sound, 'To Pitch (cc)', time_step, pitch_floor, pitch_ceiling)
       else:
              raise ValueError('Argument for @pitch_type not recognized!')
       
       mean_pitch  = call(pitch, 'Get mean', min_time, max_time, unit)
       return mean_pitch

def get_pitch_attributes(sound, pitch_type='preferred', time_step=0.001, min_time=0., max_time=0.,
                         pitch_floor=50., pitch_ceiling=800., unit='Hertz',
                         interpolation_method='Parabolic', return_values=False,
                         replacement_for_nan=0.): # estava em 75 e 600 originalmente e Parabolic no interpolation
    """
    Function to get pitch attributes such as minimum pitch, maximum pitch, mean pitch, and
    standard deviation of pitch.

    :param (parselmouth.Sound) sound: sound waveform
    :param (str) pitch_type: the type of pitch analysis to be performed; values include 'preferred'
           optimized for speech based on auto-correlation method, and 'cc' for performing acoustic
           periodicity detection based on cross-correlation method
           NOTE: Praat also includes an option for type 'ac', a variation of 'preferred' that
           requires several more parameters. We are not including this for simplification.
    :param (float) time_step: the measurement interval (frame duration), in seconds (default: 0.)
           NOTE: The default 0. value corresponds to a time step of 0.75 / pitch floor
    :param (float) min_time: minimum time value considered for time range (t1, t2) (default: 0.)
    :param (float) max_time: maximum time value considered for time range (t1, t2) (default: 0.)
           NOTE: If max_time <= min_time, the entire time domain is considered
    :param (float) pitch_floor: minimum pitch (default: 75.)
    :param (float) pitch_ceiling: maximum pitch (default: 600.)
    :param (str) unit: units of the result, 'Hertz' or 'Bark' (default: 'Hertz)
    :param (str) interpolation_method: method of sampling new data points with a discrete set of
           known data points, 'None' or 'Parabolic' (default: 'Parabolic')
    :param (bool) return_values: whether to return a continuous list of pitch values from all frames
           or not
    :param (float) replacement_for_nan: a float number that will represent frames with NaN values
    :return: (a dictionary of mentioned attributes, a list of pitch values OR None)
    """
    
    attributes = dict()

    min_time = sound.xmin
    max_time = sound.xmax

    if pitch_type == 'preferred':
       pitch = call(sound, 'To Pitch', time_step, pitch_floor, pitch_ceiling)
    elif pitch_type == 'cc':
       pitch = call(sound, 'To Pitch (cc)', time_step, pitch_floor, pitch_ceiling)
    else:
       raise ValueError('Argument for @pitch_type not recognized!')

    num_voiced_frames = call(pitch, "Count voiced frames")
    
    if num_voiced_frames == 0:
       #print("NO VOICE FRAMES WERE FOUND")
       pitch = call(sound, 'To Pitch (cc)', time_step, pitch_floor, pitch_ceiling, 1, 0.03, 0.2, 0.01, 0.4, 0.14, pitch_ceiling)
    #print(f"Number of voiced frames: {num_voiced_frames}")

    num_frames = call(pitch, "Get number of frames")
    pitch_values = [call(pitch, "Get value in frame", i, "Hertz") for i in range(1, num_frames+1)]

    min_pitch = call(pitch, 'Get minimum',min_time, max_time,unit,interpolation_method)
    max_pitch = call(pitch, 'Get maximum',min_time, max_time,unit,interpolation_method)

    attributes['f0_range'] = max_pitch - min_pitch

    mean_pitch  = call(pitch, 'Get mean',min_time, max_time,unit)

    attributes['f0_maxavg_diff'] = max_pitch - mean_pitch
    attributes['f0_avgmin_diff'] = mean_pitch - min_pitch

    return attributes, pitch_values, mean_pitch