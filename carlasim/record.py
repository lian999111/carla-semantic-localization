"""Implements data recorder"""

import pickle
import os


class SequentialRecorder(object):
    """
    Class for sequential data recording. 
    """

    def __init__(self, record_config: dict):
        """
        Constructor method.

        Input:
            record_config: Dict of toggles specifying data to be recorded.
        """
        # Use a dict as record buffer
        self.seq_data_buffer = {}

        # Init the buffer according to the record_config
        # group can be sensor names, e.g. imu, gnss, or ground truth types, e.g. pose, lane
        for group, data_toggles in record_config.items():
            # If any of data of this group is selected, set up a record buffer for it
            if self.must_record(data_toggles):
                self.seq_data_buffer[group] = self.set_up_seq_data_buffer(
                    data_toggles)

    def record_seq(self, source):
        """ Record sequential data from source at the current time step. """
        # Iterate over groups
        for group, data_buffers in self.seq_data_buffer.items():
            # Iterate over specific data
            for data_name in data_buffers:
                data_buffers[data_name].append(source[group][data_name])

    def save(self, path_to_folder, file_name):
        """ Save recorded data under designated directory as .pkl file. """
        print('Saving sequential data.')
        with open(os.path.join(path_to_folder, file_name+'.pkl'), 'wb') as f:
            pickle.dump(self.seq_data_buffer, f)

    def must_record(self, toggles):
        """ 
        Determine if recording of a group is necessary. 

        It iterates through the toggles in the dict and return True if any of the toggles is set to On (True).

        Input:
            toggles: Dict of toggles specifying which data of a group has to be recorded.
        Output:
            True if must record this group. Otherwise, False. 
        """
        for toggle_value in toggles.values():
            if toggle_value == True:
                return True
        return False

    def set_up_seq_data_buffer(self, toggles):
        """ 
        Set up sequential data buffer for data toggled on. 

        The toggles is a dictionary of format: 
            {'data1': True, 'data2', False, 'data3', True}.
        It iterates through the toggles and creates empty lists to record the coming sequential data.
        The generated buffer corresponding to the toggles above is:
            {'data1: [], 'data3': []}

        Input:
            toggles: Dict of toggles specifying which data has to be recorded.
        Output:
            buffer: Dict of lists for storing data with corresponding data name as key. 
        """
        # Initialize the dict to store data that are toggled on
        buffer = {}
        for data_name, toggle_value in toggles.items():
            # If a toggle is on, initialize a list for it
            if toggle_value:
                buffer[data_name] = []

        return buffer


class StaticAndSequentialRecorder(SequentialRecorder):
    """
    Class for static and sequential data recording.

    Internal buffers are divided into 'static' and 'seq' (sequential) parts. The 'static' part is for data that
    don't change across time. The 'static' part must be recorded using the method record_static() only once 
    while the 'seq' part  is recorded using record_seq() at each time step.
    """

    def __init__(self, record_config: dict):
        """
        Constructor method.

        Input:
            record_config: Dict of toggles specifying data to be recorded.
        """
        seq_record_config = record_config['seq']
        static_record_config = record_config['static']

        super().__init__(seq_record_config)
        self.static_data_buffer = self.set_up_static_data_buffer(
            static_record_config)

    def record_static(self, static_data_source):
        """ Record static data from source. """
        for data_name in self.static_data_buffer:
            self.static_data_buffer[data_name] = static_data_source[data_name]

    def save(self, path_to_folder, file_name):
        """ 
        Save recorded data under designated directory as .pkl file. 
        
        The buffers for static and sequential data are combined into one dictionary before saving.
        The resutling dict has 2 key-value pairs:
            1. 'static': static_data_buffer
            2. 'seq': seq_data_buffer
        """
        print('Saving static and sequential data.')
        combined_buffer = {'static': self.static_data_buffer, 'seq': self.seq_data_buffer}
        with open(os.path.join(path_to_folder, file_name+'.pkl'), 'wb') as f:
            pickle.dump(combined_buffer, f)

    def set_up_static_data_buffer(self, toggles):
        """ 
        Set up static data buffer for data toggled on. 

        The toggles is a dictionary of format: 
            {'data1': True, 'data2', False, 'data3', True}.
        It iterates through the toggles and creates empty lists to record the coming sequential data.
        The generated buffer corresponding to the toggles above is:
            {'data1: [], 'data3': []}

        Input:
            toggles: Dict of toggles specifying which static data has to be recorded.
        Output:
            buffer: Dict of Nones for storing static data with corresponding data name as key. 
        """
        # Initialize the dict to store data that are toggled on
        buffer = {}
        for data_name, toggle_value in toggles.items():
            # If a toggle is on, initialize a list for it
            if toggle_value:
                buffer[data_name] = None

        return buffer
