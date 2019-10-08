import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

    setuptools.setup(
            name="sctransfer",
            version="0.0.9",
            author="Jingshu Wang",
            author_email="wangjingshususan@gmail.com",
            description="Python part for scRNA-seq transfer learning denoising tool SAVER-X",
            long_description=long_description,
            long_description_content_type="text/markdown",
            url="https://github.com/jingshuw/sctransfer",
            packages=setuptools.find_packages(),
            python_requires='>=3.5',
            install_requires=['numpy>=1.7',
                'keras>=2.2.2',
                'tensorflow>=2.0.0',
                'h5py',
                'six>=1.10.0',
                'scikit-learn',
                'scanpy',
                'anndata',
                'pandas'
                ],
            license='GPL-3',
            classifiers=[
                'Programming Language :: Python :: 3',
                'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                'Operating System :: OS Independent'],
            )

