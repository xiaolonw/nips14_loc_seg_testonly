CXXFLAGS=-O2
LIBS=-L/home/rgirdhar/libs/opencv/lib
INC=-I/home/rgirdhar/libs/opencv/include
LDFLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui

all: grabcut

%: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $< $(LIBS) $(LDFLAGS) 

clean: 
	rm -rf grabcut
