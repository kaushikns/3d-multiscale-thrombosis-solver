APIVERSION=1
LIBVERSION=20121221

# Install directory, uncomment and change to override.
# default is '/usr/local'.
# Comment out for debian packaging.
#TARGET=/usr/local
TARGET=$(DESTDIR)/usr/

# Number of jobs during compilation
NCPU=4

include build/Makefile.build

uninstall:
	rm -f $(TARGET)/lib/libcvmlcpp.a $(TARGET)/lib/libcvmlcpp.so
	rm -f $(TARGET)/lib/libcvmlcpp.so.$(APIVERSION) $(TARGET)/lib/libcvmlcpp.so.$(APIVERSION).$(LIBVERSION)
	rm -rf $(TARGET)/include/cvmlcpp $(TARGET)/share/doc/cvmlcpp
	rm -rf $(TARGET)/share/cvmlcpp   $(TARGET)/include/omptl
	rm -f $(TARGET)/bin/voxelize $(TARGET)/bin/fix-stl

post-install:
	ln -s libcvmlcpp.so.$(APIVERSION).$(LIBVERSION) $(TARGET)/lib/libcvmlcpp.so.$(APIVERSION)
	ln -s libcvmlcpp.so.$(APIVERSION) $(TARGET)/lib/libcvmlcpp.so
