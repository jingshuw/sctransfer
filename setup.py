import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

    setuptools.setup(
            name="sctransfer",
            version="0.0.1",
            author="Jingshu Wang",
            author_email="wangjingshususan@gmail.com",
            description="Python part for scRNA-seq transfer learning denoising tool SAVER-X",
            long_description=long_description,
            long_description_content_type="text/markdown",
 #           url="https://github.com/jingshuw/transferdca",
            packages=setuptools.find_packages(),
            python_requires='>=3.5',
            install_requires=['numpy>=1.7',
                'keras>=2.0.8',
                'tensorflow',
                'h5py',
                'six>=1.10.0',
                'scikit-learn',
                'scanpy',
                'anndata',
                'pandas'
                ],
            license='MIT License',
            classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ],
            )

