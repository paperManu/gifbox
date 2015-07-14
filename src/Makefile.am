bin_PROGRAMS = gifengine

gifengine_SOURCES = \
	gifbox.cpp \
	filmPlayer.cpp \
	httpServer.cpp \
	stereoCamera.cpp \
	v4l2output.cpp

gifengine_CXXFLAGS = \
	$(AM_CPPFLAGS) \
	$(BOOST_CPPFLAGS) \
	$(OPENCV_CFLAGS)

gifengine_LDADD = \
	$(OPENCV_LIBS) \
	$(BOOST_SYSTEM_LIBS) \
	-lpthread
