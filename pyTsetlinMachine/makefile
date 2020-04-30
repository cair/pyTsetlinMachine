libTM: EmbeddingConvolutionalTsetlinMachine.c EmbeddingConvolutionalTsetlinMachine.h MultiClassConvolutionalTsetlinMachine.c MultiClassConvolutionalTsetlinMachine.h ConvolutionalTsetlinMachine.c ConvolutionalTsetlinMachine.h Tools.c Tools.h IndexedTsetlinMachine.c IndexedTsetlinMachine.h 
	gcc -O3 -ffast-math -c -Wall -Werror -fpic EmbeddingConvolutionalTsetlinMachine.c MultiClassConvolutionalTsetlinMachine.c ConvolutionalTsetlinMachine.c Tools.c IndexedTsetlinMachine.c
	gcc -shared -O3 -ffast-math -o libTM.so EmbeddingConvolutionalTsetlinMachine.o MultiClassConvolutionalTsetlinMachine.o ConvolutionalTsetlinMachine.o Tools.o IndexedTsetlinMachine.o

clean:
	rm *.o libTM.so
