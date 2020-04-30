libTM: EmbeddingTsetlinMachine.c EmbeddingTsetlinMachine.h ClassMultiClassConvolutionalTsetlinMachine.c MultiClassConvolutionalTsetlinMachine.h ConvolutionalTsetlinMachine.c ConvolutionalTsetlinMachine.h Tools.c Tools.h IndexedTsetlinMachine.c IndexedTsetlinMachine.h 
	gcc -O3 -ffast-math -c -Wall -Werror -fpic EmbeddingTsetlinMachine.c MultiClassConvolutionalTsetlinMachine.c ConvolutionalTsetlinMachine.c Tools.c IndexedTsetlinMachine.c
	gcc -shared -O3 -ffast-math -o libTM.so EmbeddingTsetlinMachine.o MultiClassConvolutionalTsetlinMachine.o ConvolutionalTsetlinMachine.o Tools.o IndexedTsetlinMachine.o

clean:
	rm *.o libTM.so
