#											   -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.68)
AC_INIT(SPLASH, 0.3.1, emmanueldurand@gmail.com)
AC_CONFIG_MACRO_DIR([m4])
AC_PROG_LIBTOOL
AM_INIT_AUTOMAKE([1.9 tar-ustar])
AC_CONFIG_HEADERS([src/config.h])

# Check for C and C++ compilers
AC_PROG_CXX
AC_PROG_CC_C_O
AC_PROG_OBJCXX

CC="gcc"
CXX="g++"
CXXFLAGS="$CXXFLAGS -Wall -std=c++11"

export PKG_CONFIG_PATH=$srcdir/pkgconfig:/usr/local/lib/pkgconfig:/usr/lib/pkgconfig

# Check for header files
AC_HEADER_STDC

# OpenCV, for their hardware support
PKG_CHECK_MODULES([OPENCV], [opencv >= 3.0],
    AC_DEFINE([HAVE_OPENCV], [1], [Defined to 1 if OpenCV is detected]),
    AC_DEFINE([HAVE_OPENCV], [0], [Defined to 1 if OpenCV is detected]))
AM_CONDITIONAL(HAVE_OPENCV, test -n "$OPENCV_LIBS")

PKG_CHECK_MODULES([FREENECT2], [freenect2])
if test "x${have_freenect2}" = "xfalse" ; then
    AC_MSG_ERROR([Missing libfreenect2])
fi

# Boost:
BOOST_REQUIRE([1.50])
BOOST_ASIO
BOOST_SYSTEM

AC_CONFIG_FILES([
	Makefile
	src/Makefile
    tools/Makefile
	])
AC_OUTPUT
