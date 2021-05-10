all: libanalysis.so irlexer fast
libanalysis.so:
	cd ../analyzer && mkdir -p build && cd build && cmake .. && make && cp libanalysis.so ../../bin
irlexer:
	cd ../irlexer && go build && cp irlexer ../bin
fast:
	cd ../fastBPE && g++ fastBPE/main.cc -O3 -std=c++17 -o ../bin/fast -lpthread
clean:
	rm -rf libanalysis.so irlexer fast ../irlexer/irlexer ../irlexer/go.sum ../analyzer/build