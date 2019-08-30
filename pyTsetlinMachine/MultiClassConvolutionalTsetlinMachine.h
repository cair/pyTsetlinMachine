/*

Copyright (c) 2019 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
https://arxiv.org/abs/1905.09688

*/

#include "ConvolutionalTsetlinMachine.h"

struct MultiClassTsetlinMachine {
	int number_of_classes;
	struct TsetlinMachine **tsetlin_machines;

	int number_of_patches;
	int number_of_ta_chunks;
	int number_of_state_bits;
};

struct MultiClassTsetlinMachine *CreateMultiClassTsetlinMachine(int number_of_classes, int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, int boost_true_positive_feedback);

void mc_tm_initialize(struct MultiClassTsetlinMachine *mc_tm);

void mc_tm_destroy(struct MultiClassTsetlinMachine *mc_tm);

void mc_tm_initialize_random_streams(struct MultiClassTsetlinMachine *mc_tm, float s);

void mc_tm_predict(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int number_of_examples);

void mc_tm_fit(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int y[], int number_of_examples, int epochs);

void mc_tm_get_state(struct MultiClassTsetlinMachine *mc_tm, int class, unsigned int *ta_state);

void mc_tm_set_state(struct MultiClassTsetlinMachine *mc_tm, int class, unsigned int *ta_state);

int mc_tm_ta_state(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, int ta);

int mc_tm_ta_action(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, int ta);
