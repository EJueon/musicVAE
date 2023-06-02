from typing import Dict, List

import sys
import os
import numpy as np
from pretty_midi import PrettyMIDI, Note, Instrument
from tqdm import tqdm

# custom
from utils import load_files


class MIDIConversionError(Exception):
    pass


# Reference : https://github.com/magenta/note-seq/blob/55e4432a6686cec84b392c2290d4c2a1d040675c/note_seq/drums_encoder_decoder.py#LL23C1-L51C1
DEFAULT_DRUM_TYPE_PITCHES = [
    # kick drum
    [36, 35],
    # snare drum
    [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85],
    # closed hi-hat
    [42, 44, 54, 68, 69, 70, 71, 73, 78, 80, 22],
    # open hi-hat
    [46, 67, 72, 74, 79, 81, 26],
    # low tom
    [45, 29, 41, 43, 61, 64, 84],
    # mid tom
    [48, 47, 60, 63, 77, 86, 87],
    # high tom
    [50, 30, 62, 76, 83],
    # crash cymbal
    [49, 52, 55, 57, 58],
    # ride cymbal
    [51, 53, 59, 82],
]
NUM_CLASSES = len(DEFAULT_DRUM_TYPE_PITCHES)
QUANTIZE_CUTOFF = 0.5

def get_num_classes():
    return NUM_CLASSES

def get_drum_classes(reverse=False) -> Dict:
    """Get drum pitch class with mapping

    Returns:
        Dict: drum classes map
    """
    pitch_class_map = {}
    for i, pitches in enumerate(DEFAULT_DRUM_TYPE_PITCHES):
        if not reverse:
            pitch_class_map.update({p: i for p in pitches})
        else:
            pitch_class_map.update({i: pitches})
    return pitch_class_map


def quantize_to_step(unquantized_time: np.float64, fs: np.float64):
    """Quantize time for fs

    Args:
        unquantized_time (np.float64): Seconds to quantize
        fs (np.float64): Sampling Frequency

    Returns:
        int: quantized step
    """
    unquantized_steps = unquantized_time * fs
    return int(unquantized_steps + (1 - QUANTIZE_CUTOFF))


# reference : https://github.com/magenta/note-seq/blob/55e4432a6686cec84b392c2290d4c2a1d040675c/note_seq/drums_encoder_decoder.py#L96
def encode_bin_to_dec(roll: np.ndarray):
    """Encoding the array of NUM_CLASSES into decimal format.
       (The array is treated as binary numbers.)

    Args:
        roll (np.ndarray): quantized binary roll sequence

    Returns:
        np.ndarray: quantized decimal roll sequence
    """
    new_roll = np.zeros((roll.shape[0],))
    for i, array in enumerate(roll):
        new_roll[i] = int("".join(list(map(lambda x: str(int(x)), array))), 2)
    return new_roll


# reference : https://github.com/craffel/pretty-midi/blob/0f5c1ba8eded15fd41b961d6a97164176122a3d3/pretty_midi/instrument.py#L77
def get_drum_roll(notes: Note, fs: np.float64, times: List):
    """Quantize and transform one hot classes on note sequence

    Args:
        notes (Note): note sequence
        fs (np.float64): sample frequency
        start_time (np.float64): start time of note sequence

    Returns:
        np.ndarray : quantized one hot sequence
    """
    # Get the start time of the last event
    start_time = times[0]
    # Get the end time of the last event
    end_time = times[1]
    
    pitch_class_map = get_drum_classes()
    
    # Allocate a matrix of zeros - we will add in as we go
    drum_roll = np.zeros((quantize_to_step((end_time - start_time), fs) + 1, NUM_CLASSES))
    for note in notes:
        if not pitch_class_map.get(note.pitch):
            continue
        
        start_step = quantize_to_step((note.start - start_time), fs)
        end_step = quantize_to_step((note.end - start_time), fs)
        drum_roll[start_step:end_step+1, pitch_class_map[note.pitch]] = 1
    return encode_bin_to_dec(drum_roll)


def preprocess(dir_path: str, extensions: List[str] = []):
    """Preprocess MIDI to quantized one hot sequence data

    Args:
        dir_path (str): Dataset directory path
        extensions (List[str], optional): File extensions. Defaults to [].

    Raises:
        MIDIConversionError: decoding error

    Returns:
        List : List of data
    """

    data = []
    file_paths = list(load_files(dir_path, extensions))
    for file_path in tqdm(file_paths):
        try:
            pm = PrettyMIDI(file_path)
            if not isinstance(pm, PrettyMIDI):
                raise MIDIConversionError("Midi Decoding Error %s : %s" % (sys.exc_info()[0], sys.exc_info()[1]))

            instrument = pm.instruments[0] 
            # calculate start & end time
            end_time = instrument.get_end_time()
            times = pm.get_onsets()
            if times[-1] > end_time:
                end_time = times[-1]
            times = [times[0], end_time]
            
            # calculate sample frequency
            beats = pm.get_beats()
            fs = 1. / ((beats[1] - beats[0]) / 4) # 4/4 

            if instrument.is_drum:
                drum_roll = get_drum_roll(instrument.notes, fs, times)
                data.append((os.path.basename(file_path), drum_roll))

        except Exception as e:
            print(f"Error file: {file_path}, {e}")
            continue

    return data
