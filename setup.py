from setuptools import find_packages, setup
from typing import List

HYPEN_DOT_E = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
        This function will get the requirements.txt file
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        
        if HYPEN_DOT_E in requirements:
            requirements.remove(HYPEN_DOT_E)
    return requirements
        

setup(
name = 'MLProject',
author = 'Radhika',
author_email = 'radhikamaheshwari26@gmail.com',
version = '0.0.1',
install_requires = get_requirements('requirements.txt')
)