all: 2_concatjson 5_buildvocab
2_concatjson: 2_concatjson.cc
	g++ 2_concatjson.cc -o 2_concatjson -O3 -fopenmp -std=c++17
5_buildvocab: 5_buildvocab.cc
	g++ 5_buildvocab.cc -o 5_buildvocab -O3 -std=c++17
clean:
	rm -rf 2_concatjson 5_buildvocab
