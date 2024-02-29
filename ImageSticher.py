import numpy as np
import h5py
from scipy import ndimage
from NexusWritter import NexusWritter


class Image_stitcher():

    def __init__(self, list_prj, list_path_h5_prj, list_flats, list_path_h5_flats, list_darks, list_path_h5_darks, path_rot_angles, output_path):
        """
        The achievable FOV is determined by the combination of the detector size (number of pixels)
        and the applied magnification, restricting either the FOV or relaxing spatial resolving power,
        as they are inversely correlated. Consequently, μCT is applicable to smaller objects or a (sub-)section of an
        object larger than the FOV. For any object that exceeds the FOV of the imaging system, alternative approaches
        for extending the FOV have to be considered if the whole sample is to be imaged at high resolution. This script performs the horizontal image stitching of
        projections from tomography acquisition done at different offset positions from the rotation axis. This projection stitching
        script returns an extended projection, which can then be used to reconstruct the full large volume.

        :param list_prj: List of projection data files.
        :param list_path_h5_prj: List of paths to projection HDF5 datasets.
        :param list_flats: List of flat field data files.
        :param list_path_h5_flats: List of paths to flat field HDF5 datasets.
        :param list_darks: List of dark field data files.
        :param list_path_h5_darks: List of paths to dark field HDF5 datasets.
        :param path_rot_angles: Path to rotation angles dataset.
        :param output_path: Path to output directory.

        """

        self.list_prj = list_prj
        self.list_path_h5_prj = list_path_h5_prj
        self.output_path = output_path
        self.list_flats = list_flats
        self.list_darks = list_darks
        self.list_path_h5_flats = list_path_h5_flats
        self.list_path_h5_darks = list_path_h5_darks
        self.path_rot_angles = path_rot_angles

        self._data_validation()
        self._data_reading()
        self._data_normalization()

    def _data_validation(self):
        """
        Validate data integrity.
        Check that the HDF5 files exists and data size are consistent.

        """
        if len(self.list_prj) != len(self.list_path_h5_prj):
            raise Exception('length of h5 paths is inconsistent with data')

        self.shape_list = []
        for h5path, h5data in zip(self.list_prj, self.list_path_h5_prj):
            with h5py.File(h5path,'r') as h5:
                self.shape_list.append(h5[h5data].shape)

        shape_list = np.array(self.shape_list)
        bool_array = np.all(shape_list==shape_list[0,2], axis = 0)

        if not bool_array[2]:
            raise Exception('height of frames are inconsistent')

    def _data_reading(self):
        """
        Read data from HDF5 files.
        """
        self.dict_projections = {}
        self.number_of_projections = []
        for i,data_prj, file_prj, file_flats, data_flats, file_darks, data_darks, \
                in zip(range(3),self.list_path_h5_prj,self.list_prj,self.list_flats,
                       self.list_path_h5_flats, self.list_darks, self.list_path_h5_darks):
            dataset_h5 = h5py.File(file_prj, 'r')
            datah5 = dataset_h5[data_prj]
            self.number_of_projections.append(datah5.shape[0])
            self.dict_projections[str(i)] ={}
            index0 = 0
            index90 = int(datah5.shape[0]/4)
            index180 = int(datah5.shape[0]/2)
            index270 = index90 +index180
            self.dict_projections[str(i)]['0'] = datah5[index0]
            self.dict_projections[str(i)]['90'] = datah5[index90]
            self.dict_projections[str(i)]['180'] = datah5[index180]
            self.dict_projections[str(i)]['270'] = datah5[index270]

            darks_h5 = h5py.File(file_darks, 'r')
            datadarks=darks_h5[data_darks][...]
            self.dict_projections[str(i)]['Dark'] = np.mean(datadarks, axis= 0)

            
            flats_h5 = h5py.File(file_flats, 'r')
            dataflats = flats_h5[data_flats][...]
            self.dict_projections[str(i)]['Flat'] = np.median(dataflats, axis= 0)

    def _data_normalization(self):
        """
        Normalize projection data.
        """
        for key_scan in list(self.dict_projections.keys()):
            for key_angle in list(self.dict_projections[key_scan].keys()):
                if (key_angle != 'Flat') or (key_angle != 'Dark') or (key_angle != 'NumProj'):
                    prj = self.dict_projections[key_scan][key_angle]
                    dark = self.dict_projections[key_scan]['Dark']
                    flat = self.dict_projections[key_scan]['Flat']
                    norm_prj = -np.log((prj - dark)/ (flat - dark))
                    self.dict_projections[key_scan][key_angle] = norm_prj


    def compute_displacements(self, search_window_width, displacement_guess):
        """

        :param search_window_width: List of search window withs, (1 value per overlap, meaning length is number of datasets -1).
        :param displacement_guess: List of initial displacement guesses (1 value per overlap, meaning length is number of datasets -1).
        :return: List of final displacements for each overlap.
        """
        displacement_list = []
        for i in range(0,len(search_window_width)):
            displacement_list.append([])
            for key_angle in list(self.dict_projections[str(i)].keys()):
                if (key_angle != 'Flat') or (key_angle != 'Dark'):

                    dataset_1 = self.dict_projections[str(i)][key_angle]
                    dataset_2 = self.dict_projections[str(i+1)][key_angle]

                    dataset_1_ROI = dataset_1[:, -search_window_width[i]:]
                    delta_range_1 = dataset_1.shape[1] - search_window_width[i]
                    dataset_2_ROI = dataset_2[:, 0:search_window_width[i+1]]

                    max_pos = []
                    displacement_array = []

                    for num1, column_in_1 in enumerate(dataset_1_ROI.T):
                        correlation_list = []
                        num_true_1 = num1 + delta_range_1
                        for num2, column_in_2 in enumerate(dataset_2_ROI.T):
                            corr_value = np.corrcoef(column_in_1, column_in_2)[0, 1]
                            correlation_list.append(corr_value)

                        max_value = np.max(correlation_list)

                        if max_value >= 0.2:
                            position = np.where(correlation_list == max_value)[0]

                            displacement = num_true_1 - position

                            displacement_error = abs(displacement - displacement_guess[i]) / displacement_guess[i]

                            max_pos.append([max_value, num_true_1, position])

                            if displacement_error < 0.05:
                                displacement_array.append(displacement)

                        if len(displacement_array) == 0:
                            final_displacement = np.NAN
                        else:
                            final_displacement = int(np.median(displacement_array))

                        displacement_list[i].append(final_displacement)

        self.list_final_displacement = []
        for list_i in displacement_list:
            self.list_final_displacement.append(int(np.median(list_i)))


        return self.list_final_displacement

    def _overlap_images(self, list_proj):
        """
        Overlap images to stitch them together.

        :param list_proj: List of projections to be stitched.
        :return: Stitched "extended" projection.
        """
        for i, displacement in enumerate((self.list_final_displacement)):
            dataset_1 = list_proj[i]
            dataset_2 = list_proj[i+1]
            if i == 0:
                R = dataset_1
                d = 0
            size_new_proj = int(d + displacement) + dataset_2.shape[1]
            overlap = (dataset_2.shape[1] + R.shape[1]) - size_new_proj
            position_middle = int(np.ceil(R.shape[1] - overlap / 2))
            output_proj = np.zeros((R.shape[0], size_new_proj))
            output_proj[:, :position_middle] = R[:, :position_middle]
            output_proj[:, position_middle:] = dataset_2[:, int(np.ceil(overlap / 2)):]
            section_to_filter = output_proj[:, position_middle - 25:position_middle + 25]
            filtered_section = ndimage.median_filter(section_to_filter, size=5)
            list_1 = np.arange(0, 25) * (1.0 / 25)
            list_2 = np.arange(25, -1, -1) * (1.0 / 25)
            list_positions = np.concatenate((list_1, list_2))
            for i in range(-25, 25):
                output_proj[:, position_middle + i] = list_positions[i + 25] * filtered_section[:, i + 25] + (
                            1 - list_positions[i + 25]) * section_to_filter[:, i + 25]

            R = output_proj
            d = d + displacement

        return R

    def _find_closest_values(self, list, target):
        """
        Find the closest values in a list to a target value.
        :param list: List of values.
        :param target: Target value.
        :return: List of closest values.
        """
        closest_values = []
        closest_difference = float('inf')  # Initialize with positive infinity

        for value in list:
            difference = abs(value - target)
            if difference < closest_difference:
                closest_values = [value]  # Found a closer value, reset the list
                closest_difference = difference
            elif difference == closest_difference:
                closest_values.append(value)  # Found another value with the same difference

        return closest_values



    def GenerateNX(self):
        """
        Generate NeXus file.
        """

        nx_array = np.zeros((self.number_of_projections[-1]+2,self.shape_list[0][0], self.shape[0][1]), dtype=np.uint16)
        image_key_list = []
        rotation_angle = []

        nx_w = NexusWritter(f'{self.output_path}/Stitching.nx')
        nx_w.set_scan_params_from_hdf5(list_h5[0])
        dptr = nx_w.nx_detector.create_dataset('data', chunks=True, shape=(self.number_of_projections[-1], nx_array.shape[1], nx_array.shape[2]), dtype='uint16')

        with h5py.File(self.list_darks[-1],'r') as h5:
            dark = h5[self.list_path_h5_darks][0]
            image_key_list.append(2)
            rotation_angle.append(0)
            list_darks = []
            list_darks.append(dark)
            for scan, path in zip(reversed(self.list_darks[:-2]), reversed(self.list_path_h5_darks[:-2])):
                with h5py.File(scan, 'r') as h5:
                    list_darks.append(h5[path][0])
            nx_array[0] = self._overlap_images(list_darks)

        with h5py.File(self.list_flats[-1],'r') as h5:
            flat = h5[self.list_path_h5_flats][0]
            image_key_list.append(1)
            rotation_angle.append(0)
            list_flats = []
            list_flats.append(flat)
            for scan, path in zip(reversed(self.list_flats[:-2]), reversed(self.list_path_h5_flats[:-2])):
                with h5py.File(scan, 'r') as h5:
                    list_flats.append(h5[path][0])
            nx_array[1] = self._overlap_images(list_flats)


        with h5py.File(self.list_prj[-1],'r') as h5:
            rot_angles = h5[path_rot_angles][:]

            for i, search_angle in enumerate(rot_angles):
                image_key_list.append(0)
                rotation_angle.append(search_angle)
                list_prj = []
                list_prj.append(list_path_h5[-1][i])
                for scan, path in zip(reversed(self.list_prj[:-2]), reversed(self.list_path_h5_prj[:-2])):
                    with h5py.File(scan, 'r') as h5:
                        angle_list = h5[self.path_rot_angles][:]
                        position = np.where(self._find_closest_values(angle_list, search_angle)[0])
                        list_prj.append(h5[path][position])

            nx_array[i +2] = self._overlap_images(list_prj)


            nx_w.nx_detector.create_dataset('image_key_control', data=image_key_list, chunks=True, maxshape=(None,))
            nx_w.nx_detector.create_dataset('image_key', data=image_key_list, chunks=True, maxshape=(None,))
            nx_w.nx_sample.create_dataset('rotation_angle', data=rotation_angle, chunks=True, maxshape=(None,))
            dptr[:] = nx_array[:]
            nx_w.save_nx_file()


if __name__ == '__main__':

    #Give the paths to the datasets with the projections to be stitched, as path_1, path_2 .... path_n.
    path_1 = 'W:\\Data\\IHMI1453_Data\\IHMI1453_Data_doughnut_battery_2_ystage02\\IHMI1453_Data_doughnut_battery_2_ystage02.h5'
    path_2 = 'W:\\Data\\IHMI1453_Data\\IHMI1453_Data_doughnut_battery_2_ystage02\\IHMI1453_Data_doughnut_battery_2_ystage02.h5'
    path_3 = 'W:\\Data\\IHMI1453_Data\\IHMI1453_Data_doughnut_battery_2_ystage02\\IHMI1453_Data_doughnut_battery_2_ystage02.h5'

    # Put the above paths into list_h5 = [path_1, path_2, .... path_n].
    list_h5 = [path_1, path_2, path_3]
    # Give the path to the projections in the HDF5 file according to your format, as list_path_h5 = ['your'/'format']*the number of datasets to be stitched.
    list_path_h5 = ['4.1/measurement/pcolinux']*3

    # Give the path to the dark field images in the HDF5 file according to your format, as list_darks_h5 = ['your'/'format']*the number of datasets to be stitched.
    list_darks_h5 = ['2.1/measurement/pcolinux']*3
    # Give the path to the flat field images in the HDF5 file according to your format, as list_flats_h5 = ['your'/'format']*the number of datasets to be stitched.
    list_flats_h5 = ['3.1/measurement/pcolinux']*3

    # Give the path to the rotation angles in the HDF5 file according to your format, as path_rot_angles = ['your'/'format'] (make sure it is the one for the last scan with the largest amount of projections).
    path_rot_angles = '4.1/measurement/mrsrot'

    # Give the path where the NeXus file should be saved.
    output_path = 'W:\\Data\\stitching_code_test\\'

    #search_window_list = [] is a list containing the sizes of the search window (based on how big the poverlap is). The length of the list is based on how many overlaps there are, meaning length of list_h5 -1.
    search_window_list = [600]*2
    # list_displacement_guess = [] is a list containing guesses of at which pixel position of the first frame the overlap with the second frame begins. The length of the list is based on how many overlaps there are, meaning length of list_h5 -1.
    list_displacement_guess = [2020]*2

    # Initialize the Image_stitcher class with the specified parameters.
    battery_ext_proj = Image_stitcher(list_h5,list_path_h5, list_h5, list_flats_h5, list_h5, list_darks_h5, path_rot_angles, output_path)
    # Compute displacements between projections.
    displacement = battery_ext_proj.compute_displacements(search_window_list, list_displacement_guess)
    # Generate a NeXus file incorporating the stitched projections.
    battery_ext_proj.GenerateNX()





