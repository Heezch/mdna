from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext

compile_args = ["-g", "-std=c++21", "-stdlib=libc++"]
cmdclass = {}
ext_modules = []

cmdclass.update({"build_ext": build_ext})
ext_modules += cythonize(
    [
        Extension(
            "pylk._writhemap_cython",
            ["pylk/_writhemap_cython.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ]
)
ext_modules += cythonize(
    [
        Extension(
            "pylk.linkingnumber_cython",
            ["pylk/linkingnumber_cython.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ]
)

# ~ ext_modules+=cythonize("pylk/cythonWM.pyx"),include_dirs=[numpy.get_include()]
# ~ compile_args = ['-g', '-std=c++17', '-stdlib=libc++']

setup(
    name="PyLk",
    version="0.1.0",
    description="A package to calculate linking properties of polymer configurations",
    url="https://github.com/eskoruppa/PyLk",
    author="Enrico Skoruppa",
    author_email="esk dot phys at gmail dot com",
    license="MIT",
    packages=["pylk"],
    install_requires=["numpy", "numba", "cython", "scipy"],
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    compile_args=compile_args,
)
