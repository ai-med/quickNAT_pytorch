"""
Contains commong functions useful throughout the application
"""
import os


def create_if_not(path):
    """
    Creates a folder at the given path if one doesnt exist before
    ===

    :param path: destination to check for existense
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
