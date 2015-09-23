bin_PROGRAMS = gifengine

gifengine_SOURCES = \
	gifbox.cpp \
	filmPlayer.cpp \
	httpServer.cpp \
	layerMerger.cpp \
    rgbdCamera.cpp \
	v4l2output.cpp

gifengine_CXXFLAGS = \
	$(AM_CPPFLAGS) \
	$(BOOST_CPPFLAGS) \
    $(FREENECT_CFLAGS) \
	$(OPENCV_CFLAGS)

gifengine_LDADD = \
	$(OPENCV_LIBS) \
    $(FREENECT_LIBS) \
	$(BOOST_SYSTEM_LIBS) \
	-lpthread
