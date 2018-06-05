from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import platform

libpointing = '../../../'

system = platform.system()

if system == 'Darwin':
    ext_modules = [Extension(
                        "pylibpointing",
                        ["pylibpointing.pyx"],
                        language="c++",
                        libraries=['pointing'],
                        include_dirs=[libpointing],
                        library_dirs=["./",libpointing+"pointing"],
                        extra_compile_args=["-stdlib=libc++", "-mmacosx-version-min=10.7", "-std=c++11"],
                        extra_link_args=["-mmacosx-version-min=10.7", "-framework", "CoreGraphics"],
                        )]
elif system == 'Linux':
    ext_modules = [Extension(
                         "pylibpointing",
                         ["pylibpointing.pyx"],
                         language="c++",
                         libraries=['pointing', "stdc++", "udev", "X11", "Xrandr"],
                         include_dirs=[
                            libpointing
                            ],
                         library_dirs=["./"]
                         )]
elif system == 'Windows':
    ext_modules = [Extension(
                         "pylibpointing",
                         ["pylibpointing.pyx"],
                         language="c++",
                         libraries=['pointing', "user32", "advapi32", "setupapi", "hid"],
                         include_dirs=[
                            libpointing, 'C:/Program Files (x86)/Windows Kits/10/Include/10.0.10586.0/ucrt'
                            ],
                         library_dirs=["./", 'C:/Program Files (x86)/Windows Kits/10/Lib/10.0.10586.0/ucrt/x64']
                         )]
else:
    ext_modules = [Extension(
                         "pylibpointing",
                         ["pylibpointing.pyx"],
                         language="c++",
                         libraries=['pointing', "stdc++"],
                         include_dirs=[
                            libpointing
                            ],
                         library_dirs=["./"]
                         )]

setup(
  name='pylibpointing',
  cmdclass={'build_ext': build_ext},
  ext_modules=ext_modules
)
