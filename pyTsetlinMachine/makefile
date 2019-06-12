libTM: MultiClassConvolutionalTsetlinMachine.c MultiClassConvolutionalTsetlinMachine.h ConvolutionalTsetlinMachine.c ConvolutionalTsetlinMachine.h Tools.c Tools.h
	gcc -O3 -ffast-math -c -Wall -Werror -fpic MultiClassConvolutionalTsetlinMachine.c ConvolutionalTsetlinMachine.c Tools.c
	gcc -shared -O3 -ffast-math -o libTM.so MultiClassConvolutionalTsetlinMachine.o ConvolutionalTsetlinMachine.o Tools.o

clean:
	rm *.o libTM.so
