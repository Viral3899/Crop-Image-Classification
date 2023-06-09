import os
import yaml
from box.exceptions import BoxValueError
from Crop_Image_Classification import logger
import json
import joblib
import base64
from typing import Any
from box import Box,ConfigBox
from pathlib import Path
from ensure import ensure_annotations





@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    This function reads a YAML file from a given path and returns its contents as a ConfigBox object,
    while also logging any errors that occur.

    :param path_to_yaml: This is a parameter of type Path that represents the path to the YAML file that
    needs to be read
    :type path_to_yaml: Path
    :return: a `ConfigBox` object that contains the data loaded from a YAML file located at the
    specified path.
    """

    try:
        yaml_data = yaml.safe_load(open(path_to_yaml))
        logger.info(f"yaml file : {path_to_yaml} loaded successfully")
        yaml_data = ConfigBox(yaml_data)
        return yaml_data
    except BoxValueError:
        logger.error("Error Occurred at ValueError: {}".format(path_to_yaml))
        raise ValueError('Yaml file is Empty or Invalid')
    except Exception as e:
        logger.error(f"Error occured {e}")
        raise e





@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    This function creates a list of directories at the specified path.

    :param path_to_directories: This is a parameter of type list that represents the path to the directories
    that need to be created.
    :type path_to_directories: list
    :param verbose: This is a parameter of type bool that represents whether or not to log the creation of
    the directories.
    :type verbose: bool
    :return: None
    """

    for directory in path_to_directories:
        try:
            os.makedirs(directory)
            if verbose:
                logger.info(f"Directory {directory} created successfully")
        except FileExistsError:
            if verbose:
                logger.info(f"Directory {directory} already exists")





import json

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    This function saves a dictionary to a JSON file at the specified path.

    :param path: This is a parameter of type Path that represents the path to the JSON file that
    needs to be saved.
    :type path: Path
    :param data: This is a parameter of type dict that represents the dictionary that needs to be saved.
    :type data: dict
    :return: None
    """

    # Check if the JSON file exists
    if path.is_file():
        # Load existing data from the JSON file if it is not empty
        with open(path, 'r') as infile:
            file_content = infile.read()
            existing_data = json.loads(file_content) if file_content else {}
        
        # Update the existing data with new keys and values
        existing_data.update(data)
        
        # Save the updated data back to the JSON file
        with open(path, 'w') as outfile:
            json.dump(existing_data, outfile, indent=4)
            
        logger.info(f"JSON file updated at {path}")
    else:
        # If the JSON file doesn't exist, save the data as a new file
        with open(path, 'w') as outfile:
            json.dump(data, outfile, indent=4)
            
        logger.info(f"JSON file saved at {path}")





@ensure_annotations
def load_json(path: Path):
    """
    This function loads a JSON file at the specified path and returns its contents as a dictionary.

    :param path: This is a parameter of type Path that represents the path to the JSON file that
    needs to be loaded.
    :type path: Path
    :return: a dictionary that contains the data loaded from a JSON file located at the specified path.
    """

    with open(path) as json_file:
        data = json.load(json_file)

    logger.info(f"json file loaded successfully from {path}")
    return ConfigBox(data)





@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    This function saves binary data to a specified file path and logs the action.

    :param data: The data that needs to be saved in binary format
    :type data: Any
    :param path: The path parameter is a Path object that represents the location where the binary file
    will be saved. It specifies the file path and name of the binary file
    :type path: Path
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at {path}")





@ensure_annotations
def get_file_size(path: Path) -> str:
    """
    This function takes a file path as input and returns the size of the file in kilobytes (KB) rounded
    to the nearest integer.
    
    :param path: The parameter "path" is of type Path, which is a class in the Python standard library's
    "pathlib" module. It represents a path to a file or directory on the file system
    :type path: Path
    :return: a string that represents the size of a file in kilobytes (KB), rounded to the nearest
    integer. The string includes the size in KB and the "~" symbol to indicate that the size is
    approximate.
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"




def decodeImage(imgString,filename):
    """
    This function decodes a base64 encoded image string and saves it as a file with the given filename.
    
    :param imgString: The imgString parameter is a string containing image data encoded in base64 format
    :param filename: The filename parameter is a string that represents the name of the file that the
    decoded image data will be saved to
    """
    imagedata = base64.b64decode(imgString)
    with open(filename,'wb') as f:
        f.write(imagedata)
        f.close()

def encodeImageintobase64(croppedImagePath):
    """
    This function encodes an image file into base16 format using Python's base64 module.
    
    :param croppedImagePath: The path to the image file that needs to be encoded into base64 format
    :return: the base16 encoded version of the image file located at the path specified by the
    `croppedImagePath` parameter.
    """
    with open(croppedImagePath,'rb') as f:
        return base64.b16encode(f.read())



