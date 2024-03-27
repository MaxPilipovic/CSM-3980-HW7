all: Main Main2 Main3

Main: Main.cu
	nvcc Main.cu

Main2: Main2.cu
	nvcc Main2.cu

Main3: Main3.cu
	nvcc Main3.cu

clean:
	rm -f Main Main2 Main3

test: all