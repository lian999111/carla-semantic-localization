# Implementation of data recorder

import pickle
import os


class Recorder(object):
    """
    Class for sensor data and ground truth recording. 
    """

    def __init__(self, sensor_data_source: dict, gt_data_source: dict, record_config: dict):
        """
        Constructor method.

        Input:
            sensor_data_source: Source from where sensor data can be reached.
            gt_data_source: Source from where ground truth data can be reached.
            record_config: Dict specifying data to be recorded.
        """
        # Sources
        self.sensor_data_source = sensor_data_source
        self.gt_data_source = gt_data_source
        # Buffers
        self.sensor_record_buffer = {}
        self.gt_record_buffer = {"static": {}, "seq": {}}   # divide into static nd sequential parts

        # Sensor data
        sensor_record_config = record_config['sensor']
        # Init sub-dicts for sensors and data chosen to be recorded
        for sensor_name, data_toggles in sensor_record_config.items():
            # If any of data of this sensor is selected, set up a record buffer for it
            if self.must_record(data_toggles):
                self.sensor_record_buffer[sensor_name] = self.set_up_record_buffer(
                    data_toggles)

        # Sequential ground truth
        gt_record_config = record_config['gt']
        for gt_type_name, data_toggles in gt_record_config.items():
            # If any of data of this sequential ground truth type is selected, set up a record buffer for it
            if self.must_record(data_toggles):
                self.gt_record_buffer['seq'][gt_type_name] = self.set_up_record_buffer(
                    data_toggles)

    def record_current_step(self):
        """ Record data at current simulation step. """
        # Iterate over sensors
        for sensor_name, record_buffer in self.sensor_record_buffer.items():
            # Iterate over specific data
            for data_name in record_buffer:
                record_buffer[data_name].append(
                    self.sensor_data_source[sensor_name][data_name])

        seq_gt_data_source = self.gt_data_source['seq']
        # Iterate over types of sequential ground truth data
        for gt_type_name, record_buffer in self.gt_record_buffer['seq'].items():
            # Iterate over specific data
            for data_name in record_buffer:
                record_buffer[data_name].append(
                    seq_gt_data_source[gt_type_name][data_name])

    def save_data(self, path_to_folder):
        """ Save recorded data under designated directory. """
        with open(os.path.join(path_to_folder, 'sensor_data.pkl'), 'wb') as f:
            pickle.dump(self.sensor_record_buffer, f)
        with open(os.path.join(path_to_folder, 'gt_data.pkl'), 'wb') as f:
            pickle.dump(self.gt_record_buffer, f)


    def must_record(self, toggles):
        """ 
        Determine if recording of a sensor is necessary. 

        It iterates through the toggles in the dict and return True if any of the toggles is set to On (True).

        Input:
            toggles: Dict of toggles specifying which data of a sensor has to be recorded.
        Output:
            True if must record this sensor. Otherwise, False. 
        """
        for toggle_value in toggles.values():
            if toggle_value == True:
                return True
        return False

    def set_up_record_buffer(self, toggles):
        """ 
        Set up recording buffer if necessary, given the record config of the specific sensor. 

        It iterates through the toggles in toggles and create empty lists to record corresponding data.

        Input:
            toggles: Dict of toggles specifying which data of the sensor has to be recorded.
        Output:
            record_buffer: Dict of lists for storing data with corresponding data name as key. None toggles are True. 
        """
        # Initialize the dict to store recorded data that are toggled on
        record_buffer = {}
        for data_name, toggle in toggles.items():
            # If a toggle is on, initialize a list for it
            if toggle:
                record_buffer[data_name] = []

        return record_buffer

