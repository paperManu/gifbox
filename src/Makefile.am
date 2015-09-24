bin_PROGRAMS = gifengine

gifengine_SOURCES = \
	gifbox.cpp \
	filmPlayer.cpp \
	httpServer.cpp \
	k2Camera.cpp \
	layerMerger.cpp \
	v4l2output.cpp

gifengine_CXXFLAGS = \
	$(AM_CPPFLAGS) \
	$(BOOST_CPPFLAGS) \
    $(FREENECT2_CFLAGS) \
	$(OPENCV_CFLAGS)

gifengine_LDADD = \
	$(OPENCV_LIBS) \
    $(FREENECT2_LIBS) \
	$(BOOST_SYSTEM_LIBS) \
	-lpthread
