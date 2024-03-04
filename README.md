        In microtomography, the achievable FOV is determined by the combination of the detector size 
        (number of pixels) and the applied magnification, restricting either the FOV or relaxing spatial 
        resolving power,as they are inversely correlated. Consequently, μCT is applicable to smaller 
        objects or a (sub-)section of anobject larger than the FOV. For any object that exceeds the FOV of 
        the imaging system, alternative approaches for extending the FOV have to be considered if the whole
        sample is to be imaged at high resolution. This script performs the horizontal image stitching of 
        projections from tomography acquisition done at different offset positions from the rotation axis. 
        This projection stitching script returns an extended projection, which can then be used to reconstruct 
        the full large volume.

        In the main, please define the following: 

        1. Define the paths to the datasets with the projections to be stitched, as path_1, path_2 .... path_n.
        
        2. Put the above paths into list_h5 = [path_1, path_2, .... path_n].
        
        3. Give the path to the projections in the HDF5 file according to your format, 
           as list_path_h5 = ['your'/'format']*the number of datasets to be stitched.
           Example: ['4.1/measurement/pcolinux']*3
           
        4. Give the path to the dark field images in the HDF5 file according to your format, 
        as list_darks_h5 = ['your'/'format']*the number of datasets to be stitched.
           Example: ['2.1/measurement/pcolinux']*3
           
        5. Give the path to the flat field images in the HDF5 file according to your format, 
           as list_flats_h5 = ['your'/'format']*the number of datasets to be stitched.
           Example: ['3.1/measurement/pcolinux']*3
           
        6. Give the path to the rotation angles in the HDF5 file according to your format,
           as path_rot_angles = ['your'/'format']
           (make sure it is the one for the last scan with the largest amount of projections).
           Example: '4.1/measurement/mrsrot'
           
        7. Give the path where the NeXus file should be saved.
        
        8. Define: search_window_list = [], a list containing the sizes of the search window 
           (based on how big the poverlap is). 
           The length of the list is based on how many overlaps there are, meaning length of list_h5 -1.
           
        9. Define: list_displacement_guess = [], a list containing guesses of at which pixel position
           of the first frame the overlap with the second frame begins. 
           The length of the list is based on how many overlaps there are, meaning length of list_h5 -1.

        10. Initialize the Image_stitcher class with the specified parameters.

        11. Call the compute displacements between projections function, compute_displacements.

        12. Generate a NeXus file incorporating the stitched projections by calling GenerateNX.
        
        
