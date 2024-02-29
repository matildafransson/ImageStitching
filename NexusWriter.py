import h5py
import ntpath
import os
import numpy as np


class NexusWritter:

    def __init__(self, output_path):
        """
        Init all attributes and create new nx file if nx file already exist delete and create a new one
        Call for the initialiezation of groups and subgroups and put defaults values for single parameters (not signals)

        """
        self.output_path = output_path
        self.sample_name = ntpath.basename(self.output_path)
        self.sample_name = os.path.splitext(self.sample_name)[0]

        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

        self.nx_file = h5py.File(self.output_path, 'w')
        self._init_nx_groups()
        self._init_defaults_attributes()

    def _init_nx_groups(self):
        """
        init all group and subgroup of the nx file entry / beam / instrument / detector / sample
        """
        self.nx_entry = self.nx_file.create_group(u'entry0000')
        self.nx_entry.attrs[u'NX_class'] = u'NXentry'

        self.beam = self.nx_entry.create_group(u'beam')

        self.nx_instru = self.nx_entry.create_group(u'instrument')
        self.nx_instru.attrs[u'NX_class'] = u'NXinstrument'

        self.nx_detector = self.nx_instru.create_group(u'detector')
        self.nx_detector.attrs[u'NX_class'] = u'NXdetector'

        self.nx_sample = self.nx_entry.create_group(u'sample')
        self.nx_sample.attrs[u'NX_class'] = u'NXsample'

    def _init_defaults_attributes(self):
        """
        Init all non signal parameters to defaults values
        :return:
        """
        self.beam['incident_energy'] = 50.0

        self.nx_detector['distance'] = 1.0
        self.nx_detector['field_of_view'] = "Full"
        self.nx_detector['x_magnified_pixel_size'] = 1e-06
        self.nx_detector['x_pixel_size'] = 1e-06
        self.nx_detector['y_magnified_pixel_size'] = 1e-06
        self.nx_detector['y_pixel_size'] = 1e-06

        self.nx_sample['name'] = self.sample_name

    def get_dataset_from_nx(self, param):

        if param in self.beam:
            return np.array(self.beam.get(param))

        elif param in self.nx_detector:
            return np.array(self.nx_sample.get(param))

        elif param in self.nx_sample:
            return np.array(self.nx_sample.get(param))

    def set_scan_params_from_dic(self, dic):
        """
        Set parameters (non signals) from a dictionary
        :param dic: keys of the dictionnary should be the same as the regulard nabu nx file keys
 		distance: 0.115 [float64]
 		field of view: "Full" [string]
 		x_magnified_pixel: 6.559e-06 [float64]
 		x_pixel_size:  6.5e-06  [float64]
 		x_magnified_pixel: 6.559e-06 [float64]
 		y_pixel_size:  6.5e-06  [float64]
 	    name sample_name [string]
        :return:
        """

        for key in dic:
            for attrs in self.beam:
                if key == attrs:
                    data = self.beam[key]
                    data[...] = dic[key]

            for attrs in self.nx_detector:
                if key == attrs:
                    data = self.nx_detector[key]
                    data[...] = dic[key]

            for attrs in self.nx_sample:
                if key == attrs:
                    data = self.nx_sample[key]
                    data[...] = dic[key]

    def set_scan_params_from_hdf5(self, path_h5):
        """
        Take parameters fron an hdf5 file to copy its non signal parameters to the nx file
        :param path_h5:
        :return:
        """

        p_dic = {}

        with h5py.File(path_h5, 'r') as h5_f:
            h5_entry = h5_f[list(h5_f.keys())[0]]
            h5_technique = h5_entry['technique']
            h5_scan = h5_technique['scan']
            h5_optic = h5_technique['optic']
            p_dic['distance'] = np.array(h5_scan.get('sample_detector_distance')) / 1000.0
            p_dic['incident_energy'] = float(np.array(h5_scan.get('energy')))

            scan_type = str(np.array(h5_scan.get('field_of_view')))
            if scan_type == "Full":
                p_dic['field of view'] = 'Full'
            elif scan_type == "Half":
                p_dic['field of view'] = 'Half'
            else:
                raise Exception('Unrecognised scan type')

            p_dic['x_pixel_size'] = round(np.array(h5_optic.get('magnification')) * np.array(
                h5_optic.get('sample_pixel_size')), 4) * 1e-06
            p_dic['y_pixel_size'] = round(np.array(h5_optic.get('magnification')) * np.array(
                h5_optic.get('sample_pixel_size')), 4) * 1e-06

            p_dic['x_magnified_pixel_size'] = np.array(h5_optic.get('sample_pixel_size')) * 1e-06
            p_dic['y_magnified_pixel_size'] = np.array(h5_optic.get('sample_pixel_size')) * 1e-06

            h5_f.close()

        self.set_scan_params_from_dic(p_dic)

    def add_radios_from_array(self, dic_arr):
        self.add_data_from_array(dic_arr, 0.0)

    def add_flats_from_array(self, dic_arr):
        self.add_data_from_array(dic_arr, 1.0)

    def add_darks_from_array(self, dic_arr):
        self.add_data_from_array(dic_arr, 2.0)

    def add_data_from_array(self, dic_arr, image_key):
        '''
        add signals and data images from an dictionnary if data doesn't exist create a new dataset else append the data to the current one
        image_key and image_keycontrol is automacally create if method is called from add_radios, add_flats or add_darks
        Dict
        :param dic_arr: {'data':3d np array, 'count_time': float or numpy_array [s], 'rotation_angle': ('Full' or 'Half' or numpy array) [def], 'x_translation': (float or numpy array) [mm]}
        :param image_key:

        '''
        # Data
        if isinstance(dic_arr['data'], np.ndarray):
            if 'data' in self.nx_detector.keys():
                current_size = self.nx_detector['data'].shape[0]
            else:
                current_size = 0
            new_size = dic_arr['data'].shape[0]
            new_data = dic_arr['data']
        else:
            raise Exception('data array must be an numpy ndarray')

        if 'data' in self.nx_detector.keys():
            self.nx_detector['data'].resize((current_size + new_size), axis=0)
            self.nx_detector['data'][-new_size:] = new_data
        else:
            self.nx_detector.create_dataset('data', data=new_data, chunks=True,
                                            maxshape=(None, None, None,))
        # count_time
        if 'count_time' in dic_arr.keys():
            if isinstance(dic_arr['count_time'], np.ndarray):
                if len(dic_arr['count_time']) == new_size:
                    new_data = dic_arr['count_time']
                else:
                    raise Exception('count_time length is inconsistent with size of data')

            elif isinstance(dic_arr['count_time'], float):
                new_data = np.ones(new_size) * dic_arr['count_time']
            else:
                raise Exception('count_time data type is not compatible')
        else:
            value = 0.1
            new_data = np.ones(new_size) * value

        if 'count_time' in self.nx_detector.keys():
            self.nx_detector['count_time'].resize((current_size + new_size), axis=0)
            self.nx_detector['count_time'][-new_size:] = new_data
        else:
            self.nx_detector.create_dataset('count_time', data=new_data, chunks=True,
                                            maxshape=(None,))
        # image_key
        new_data = np.ones(new_size) * image_key

        if 'image_key' in self.nx_detector.keys():
            self.nx_detector['image_key'].resize((current_size + new_size), axis=0)
            self.nx_detector['image_key'][-new_size:] = new_data
        else:
            self.nx_detector.create_dataset('image_key', data=new_data, chunks=True,
                                            maxshape=(None,))
        # image_key_control
        if 'image_key_control' in self.nx_detector.keys():
            self.nx_detector['image_key_control'].resize((current_size + new_size), axis=0)
            self.nx_detector['image_key_control'][-new_size:] = new_data
        else:
            self.nx_detector.create_dataset('image_key_control', data=new_data, chunks=True,
                                            maxshape=(None,))

        # angle
        if 'rotation_angle' in dic_arr.keys():
            if isinstance(dic_arr['rotation_angle'], np.ndarray):
                if len(dic_arr['rotation_angle']) == new_size:
                    new_data = dic_arr['rotation_angle']
                else:
                    raise Exception('rotation_angle length is inconsistent with size of data')

            elif isinstance(dic_arr['rotation_angle'], str):
                if dic_arr['rotation_angle'] in ['full', 'Full', 'FULL', '360']:
                    step = 360.0 / new_size
                    new_data = np.arange(0, 360.0, step)
                elif dic_arr['rotation_angle'] in ['half', 'Half', 'HALF', '180']:
                    step = 180.0 / new_size
                    new_data = np.arange(0, 180.0, step)
            elif isinstance(dic_arr['rotation_angle'], float):
                new_data = np.ones(new_size) * np.array(dic_arr['rotation_angle'])
            else:
                raise Exception('rotation_angle Data type is not compatible')
        else:
            if image_key == 0:
                step = 360.0 / new_size
                new_data = np.arange(0, 360.0, step)
            else:
                new_data = np.zeros(new_size)

        if 'rotation_angle' in self.nx_sample.keys():
            self.nx_sample['rotation_angle'].resize((current_size + new_size), axis=0)
            self.nx_sample['rotation_angle'][-new_size:] = new_data
        else:
            self.nx_sample.create_dataset('rotation_angle', data=new_data, chunks=True, maxshape=(None,))

        # x_translation
        if 'x_translation' in dic_arr.keys():
            if isinstance(dic_arr['x_translation'], np.ndarray):
                if len(dic_arr['x_translation']) == new_size:
                    new_data = dic_arr['x_translation']
                else:
                    raise Exception('x_translation length is inconsistent with size of data')

            elif isinstance(dic_arr['x_translation'], float):
                new_data = np.ones(new_size) * dic_arr['x_translation']
            else:
                raise Exception('x_translation Data type is not compatible')
        else:
            new_data = np.zeros(new_size)

        if 'x_translation' in self.nx_sample.keys():
            self.nx_sample['x_translation'].resize((current_size + new_size), axis=0)
            self.nx_sample['x_translation'][-new_size:] = new_data
        else:
            self.nx_sample.create_dataset('x_translation', data=new_data, chunks=True,
                                          maxshape=(None,))
        # y_translation
        if 'y_translation' in dic_arr.keys():
            if isinstance(dic_arr['y_translation'], np.ndarray):
                if len(dic_arr['y_translation']) == new_size:
                    new_data = dic_arr['y_translation']
                else:
                    raise Exception('y_translation length is inconsistent with size of data')

            elif isinstance(dic_arr['y_translation'], float):
                new_data = np.ones(new_size) * dic_arr['y_translation']
            else:
                raise Exception('y_translation Data type is not compatible')
        else:
            new_data = np.zeros(new_size)

        if 'y_translation' in self.nx_sample.keys():
            self.nx_sample['y_translation'].resize((current_size + new_size), axis=0)
            self.nx_sample['y_translation'][-new_size:] = new_data
        else:
            self.nx_sample.create_dataset('y_translation', data=new_data, chunks=True,
                                          maxshape=(None,))
        # z_translation
        if 'z_translation' in dic_arr.keys():
            if isinstance(dic_arr['z_translation'], np.ndarray):
                if len(dic_arr['z_translation']) == new_size:
                    new_data = dic_arr['z_translation']
                else:
                    raise Exception('z_translation length is inconsistent with size of data')

            elif isinstance(dic_arr['z_translation'], float):
                new_data = np.ones(new_size) * dic_arr['z_translation']
            else:
                raise Exception('z_translation Data type is not compatible')
        else:
            new_data = np.zeros(new_size)

        if 'z_translation' in self.nx_sample.keys():
            self.nx_sample['z_translation'].resize((current_size + new_size), axis=0)
            self.nx_sample['z_translation'][-new_size:] = new_data
        else:
            self.nx_sample.create_dataset('z_translation', data=new_data, chunks=True,
                                          maxshape=(None,))

    def add_radios_from_hdf5(self, path_h5, scan_number, angle_array=None):
        self.add_data_from_hdf5(path_h5, scan_number, 0.0, angle_array)

    def add_flats_from_hdf5(self, path_h5, scan_number, angle_array=None):
        self.add_data_from_hdf5(path_h5, scan_number, 1.0, angle_array)

    def add_darks_from_hdf5(self, path_h5, scan_number, angle_array=None):
        self.add_data_from_hdf5(path_h5, scan_number, 2.0, angle_array)

    def add_data_from_hdf5(self, path_h5, scan_number, image_key, angle_array=None):
        """
        Copy signals and data from hdf5 file (hdf5 in the dataset folder)
        :param path_h5: path to h5 file
        :param scan_number: scan to select : Ex "3.1"
        :param image_key: Parameters used by add_radios, add_darks, and add_flats to initiate image_key and image_key_control
        :param angle_array: If given by the user the angle_array is used instead of the measurement/mrsrot array
        :return:
        """

        with h5py.File(path_h5, 'r') as h5_f:

            h5_scan = h5_f[scan_number]
            camera_name = str(np.array(h5_scan['technique/detector/name']))
            h5_camera = h5_scan['instrument/' + camera_name]
            h5_acquisition = h5_camera['acq_parameters']

            if 'data' in self.nx_detector.keys():

                data_shape_to_add = np.array(h5_camera['data']).shape[0]
                current_size = self.nx_detector['data'].shape[0]

                data = np.array(h5_camera['data'])

                self.nx_detector['data'].resize((data_shape_to_add + current_size), axis=0)
                self.nx_detector['data'][-data_shape_to_add:] = data

                data = np.ones(data_shape_to_add) * np.array(h5_acquisition['acq_expo_time'])

                self.nx_detector['count_time'].resize((data_shape_to_add + current_size), axis=0)
                self.nx_detector['count_time'][-data_shape_to_add:] = data

                data = np.ones(data_shape_to_add) * image_key

                self.nx_detector['image_key'].resize((data_shape_to_add + current_size), axis=0)
                self.nx_detector['image_key'][-data_shape_to_add:] = data

                self.nx_detector['image_key_control'].resize((data_shape_to_add + current_size), axis=0)
                self.nx_detector['image_key_control'][-data_shape_to_add:] = data

                if angle_array == None:
                    if 'mrsrot' in h5_scan['measurement'].keys():
                        data = np.array(h5_scan['measurement/mrsrot'])
                    elif 'mhsrot' in h5_scan['measurement'].keys():
                        data = np.array(h5_scan['measurement/mhsrot'])
                    elif 'rotm' in h5_scan['measurement'].keys():
                        data = np.array(h5_scan['measurement/rotm'])
                else:
                    if isinstance(angle_array, np.ndarray):
                        data = angle_array
                    else:
                        raise Exception('Angle Array Data type is not compatible')

                self.nx_sample['rotation_angle'].resize((data_shape_to_add + current_size), axis=0)
                self.nx_sample['rotation_angle'][-data_shape_to_add:] = data

                data = (np.array(h5_scan['instrument/positioners/sx']) / 1000.0) * np.ones(data_shape_to_add)
                self.nx_sample['x_translation'].resize((data_shape_to_add + current_size), axis=0)
                self.nx_sample['x_translation'][-data_shape_to_add:] = data

                data = (np.array(h5_scan['instrument/positioners/sy']) / 1000.0) * np.ones(data_shape_to_add)
                self.nx_sample['y_translation'].resize((data_shape_to_add + current_size), axis=0)
                self.nx_sample['y_translation'][-data_shape_to_add:] = data

                data = (np.array(h5_scan['instrument/positioners/sz']) / 1000.0) * np.ones(data_shape_to_add)
                self.nx_sample['z_translation'].resize((data_shape_to_add + current_size), axis=0)
                self.nx_sample['z_translation'][-data_shape_to_add:] = data


            else:
                self.nx_detector.create_dataset('data', data=np.array(h5_camera['data']), chunks=True,
                                                maxshape=(None, None, None,))
                self.nx_detector.create_dataset('count_time',
                                                data=np.ones(self.nx_detector['data'].shape[0]) * np.array(
                                                    h5_acquisition['acq_expo_time']), chunks=True, maxshape=(None,))
                self.nx_detector.create_dataset('image_key',
                                                data=np.ones(self.nx_detector['data'].shape[0]) * image_key,
                                                chunks=True, maxshape=(None,))
                self.nx_detector.create_dataset('image_key_control',
                                                data=np.ones(self.nx_detector['data'].shape[0]) * image_key,
                                                chunks=True, maxshape=(None,))

                if angle_array == None:
                    if 'mrsrot' in h5_scan['measurement'].keys():
                        self.nx_sample.create_dataset('rotation_angle', data=np.array(h5_scan['measurement/mrsrot']),
                                                      chunks=True, maxshape=(None,))
                    elif 'mhsrot' in h5_scan['measurement'].keys():
                        self.nx_sample.create_dataset('rotation_angle', data=np.array(h5_scan['measurement/mhsrot']),
                                                      chunks=True, maxshape=(None,))
                    elif 'rotm' in h5_scan['measurement'].keys():
                        self.nx_sample.create_dataset('rotation_angle', data=np.array(h5_scan['measurement/rotm']),
                                                      chunks=True, maxshape=(None,))

                else:
                    if isinstance(angle_array, np.ndarray):
                        self.nx_sample.create_dataset('rotation_angle', data=angle_array, chunks=True, maxshape=(None,))
                    else:
                        raise Exception('Angle Array Data type is not compatible')

                self.nx_sample.create_dataset('x_translation',
                                              data=(np.array(h5_scan['instrument/positioners/sx']) / 1000.0) * np.ones(
                                                  self.nx_detector['data'].shape[0]), chunks=True, maxshape=(None,))
                self.nx_sample.create_dataset('y_translation',
                                              data=(np.array(h5_scan['instrument/positioners/sy']) / 1000.0) * np.ones(
                                                  self.nx_detector['data'].shape[0]), chunks=True, maxshape=(None,))
                self.nx_sample.create_dataset('z_translation',
                                              data=(np.array(h5_scan['instrument/positioners/sz']) / 1000.0) * np.ones(
                                                  self.nx_detector['data'].shape[0]), chunks=True, maxshape=(None,))

    def save_nx_file(self):
        """
        Save and Close the nx file
        :return:
        """
        self.nx_file.close()

    def filter_out_data(self, signal_filter, mask=None, range=[]):

        if mask == None:
            if signal_filter == 'count_time':
                data_to_filter = np.array(self.nx_detector['count_time'])
            elif signal_filter == 'image_key':
                data_to_filter = np.array(self.nx_detector['image_key'])
            elif signal_filter == 'image_key_control':
                data_to_filter = np.array(self.nx_detector['image_key_control'])
            elif signal_filter == 'rotation_angle':
                data_to_filter = np.array(self.nx_sample['image_key_control'])
            elif signal_filter == 'x_translation':
                data_to_filter = np.array(self.nx_sample['x_translation'])
            elif signal_filter == 'y_translation':
                data_to_filter = np.array(self.nx_sample['y_translation'])
            elif signal_filter == 'z_translation':
                data_to_filter = np.array(self.nx_sample['z_translation'])

            mask = np.logical_and(data_to_filter < range[0], data_to_filter > range[1])
            print(mask)