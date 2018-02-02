import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
PACKAGES = find_packages()

extensions = [Extension("SparseGroupLasso.linalg",
                        ["SparseGroupLasso/linalg.pyx"])]

# Get version and release info, which is all stored in cloudknot/version.py
ver_file = os.path.join('SparseGroupLasso', 'version.py')
with open(ver_file) as f:
    exec(f.read())

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            install_requires=REQUIRES,
            ext_modules = cythonize(extensions),
            )


if __name__ == '__main__':
    setup(**opts)
