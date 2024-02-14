from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Context aware chunking using perplexity'
LONG_DESCRIPTION = ''

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="context_aware_chunker", 
        version=VERSION,
        author="Tanishk Kithannae",
        author_email="tanishk.kithannae@wordlabs.io",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
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