import shutil
import os

def move_folders(input_folders, filetype, output_folder):
    for folder in input_folders:
        new_folder = folder.split("/")[-2]
        if not os.path.exists(f"{output_folder}/{new_folder}/"):
            os.makedirs(f"{output_folder}/{new_folder}/")
        filelist = [filename for filename in os.listdir(folder) if filetype in filename] # create list of files
        for filename in filelist:
            shutil.copyfile(f"{folder}{filename}", f"{output_folder}/{new_folder}/{filename}")    
                # if need to change filenames could have a function defined above that changes it
            

def locate_raw_drive_files(input_path='raw_data/raw_data.txt'):
    """Collects path to raw data from txt file containing plain text address
    Parameters
    ----------
    input_path : str, optional
        plain text address for raw_data files, by default 'raw_data/raw_data.txt'
    Returns
    -------
    str
        path to raw data
    """
    with open(input_path) as f:
        data_path = [line[:-1] for line in f.readlines()]
    return data_path


