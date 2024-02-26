from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "readme.md").read_text()

VERSION = '0.0.2' 
DESCRIPTION = 'Context aware chunking using perplexity'
LONG_DESCRIPTION = long_description

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="context_aware_chunker", 
        version=VERSION,
        author="Tanishk Kithannae",
        author_email="tanishk.kithannae@wordlabs.io",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type = 'text/markdown',
        url = "https://github.com/wordlabs-io/context_aware_chunker",
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        install_requires=[
            'transformers',
            'pysbd'
        ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'rag'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)