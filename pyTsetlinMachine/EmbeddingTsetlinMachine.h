/*

Copyright (c) 2020 Ole-Christoffer Granmo

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

struct EmbeddingTsetlinMachine {
	int number_of_classes;

	struct TsetlinMachine *tsetlin_machine;

	int number_of_patches;
	int number_of_ta_chunks;
	int number_of_state_bits;
};

struct EmbeddingTsetlinMachine *CreateEmbeddingTsetlinMachine(int number_of_classes, int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range, int boost_true_positive_feedback, int weighted_clauses);

void etm_initialize(struct EmbeddingTsetlinMachine *etm);

void etm_destroy(struct EmbeddingTsetlinMachine *etm);

void etm_initialize_random_streams(struct EmbeddingTsetlinMachine *etm, float s);

void etm_predict(struct EmbeddingTsetlinMachine *etm, unsigned int *X, int *y, int number_of_examples);

void etm_fit(struct EmbeddingTsetlinMachine *etm, unsigned int *X, int y[], int number_of_examples, int epochs);

void etm_get_state(struct EmbeddingTsetlinMachine *etm, unsigned int *clause_weights, unsigned int *ta_state);

void etm_set_state(struct EmbeddingTsetlinMachine *etm, unsigned int *clause_weights, unsigned int *ta_state);

int etm_ta_state(struct EmbeddingTsetlinMachine *etm, int clause, int ta);

int etm_ta_action(struct EmbeddingTsetlinMachine *etm, int clause, int ta);

void etm_transform(struct EmbeddingTsetlinMachine *etm, unsigned int *X,  unsigned int *X_transformed, int invert, int number_of_examples);

void etm_clause_configuration(struct EmbeddingTsetlinMachine *etm, int clause, unsigned int *clause_configuration);

float etm_clause_sharing(struct EmbeddingTsetlinMachine *etm, int class_1, int class_2);
