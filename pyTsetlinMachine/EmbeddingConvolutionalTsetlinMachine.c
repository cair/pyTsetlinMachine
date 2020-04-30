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

#include <stdio.h>
#include <stdlib.h>

#include "EmbeddingConvolutionalTsetlinMachine.h"

/*****************************************************/
/*** The Class Input Convolutional Tsetlin Machine ***/
/*****************************************************/

/*** Initialize Tsetlin Machine ***/
struct EmbeddingTsetlinMachine *CreateEmbeddingTsetlinMachine(int number_of_classes, int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range, int boost_true_positive_feedback, int weighted_clauses)
{
	struct EmbeddingTsetlinMachine *etm = (void *)malloc(sizeof(struct EmbeddingTsetlinMachine));

	etm->number_of_classes = number_of_classes;

	etm->tsetlin_machine = CreateTsetlinMachine(number_of_clauses, number_of_features, number_of_patches, number_of_ta_chunks, number_of_state_bits, T, s, s_range, boost_true_positive_feedback, weighted_clauses);
	
	etm->number_of_patches = number_of_patches;

	etm->number_of_ta_chunks = number_of_ta_chunks;

	etm->number_of_state_bits = number_of_state_bits;

	return etm;
}

void etm_initialize(struct EmbeddingTsetlinMachine *etm)
{
	tm_initialize(etm->tsetlin_machine);
}

void etm_destroy(struct EmbeddingTsetlinMachine *etm)
{
	tm_destroy(etm->tsetlin_machine);

	free(etm->tsetlin_machine);
}

/***********************************/
/*** Predict classes of inputs X ***/
/***********************************/

void etm_predict(struct EmbeddingTsetlinMachine *etm, unsigned int *X, int *y, int number_of_examples)
{
	int max_class;
	int max_class_sum;

	unsigned int step_size = etm->number_of_patches * etm->number_of_ta_chunks;

	unsigned int pos = 0;

	for (int l = 0; l < number_of_examples; l++) {
		// Identify class with largest output
		max_class_sum = -etm->tsetlin_machine->T;
		max_class = 0;
		for (int i = 0; i < etm->number_of_classes; i++) {	
			unsigned int feature_chunk = i / 32;
			unsigned int feature_chunk_pos = i % 32;

			unsigned int inverted_feature_chunk = (i + etm->tsetlin_machine->number_of_features/2) / 32;
			unsigned int inverted_feature_chunk_pos = (i + etm->tsetlin_machine->number_of_features/2) % 32;

			unsigned int *Xi = &X[pos];

			for (int patch = 0; patch < etm->number_of_patches; ++patch) {
				Xi[patch*etm->number_of_ta_chunks + feature_chunk] |= (1 << feature_chunk_pos);
				Xi[patch*etm->number_of_ta_chunks + inverted_feature_chunk] &= ~(1 << inverted_feature_chunk_pos);
			}

			int class_sum = tm_score(etm->tsetlin_machine, Xi);

			for (int patch = 0; patch < etm->number_of_patches; ++patch) {
				Xi[patch*etm->number_of_ta_chunks + feature_chunk] &= ~(1 << feature_chunk_pos);
				Xi[patch*etm->number_of_ta_chunks + inverted_feature_chunk] |= (1 << inverted_feature_chunk_pos);
			}
			
			if (max_class_sum < class_sum) {
				max_class_sum = class_sum;
				max_class = i;
			}
		}

		y[l] = max_class;

		pos += step_size;
	}
	
	return;
}

/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void etm_update(struct EmbeddingTsetlinMachine *etm, unsigned int *Xi, int target_class)
{
	if (1.0*rand()/((unsigned int)RAND_MAX) <= 0.5) {
		unsigned int feature_chunk = target_class / 32;
		unsigned int feature_chunk_pos = target_class % 32;

		unsigned int inverted_feature_chunk = (target_class + etm->tsetlin_machine->number_of_features/2) / 32;
		unsigned int inverted_feature_chunk_pos = (target_class + etm->tsetlin_machine->number_of_features/2) % 32;

		for (int patch = 0; patch < etm->number_of_patches; ++patch) {
			Xi[patch*etm->number_of_ta_chunks + feature_chunk] |= (1 << feature_chunk_pos);
			Xi[patch*etm->number_of_ta_chunks + inverted_feature_chunk] &= ~(1 << inverted_feature_chunk_pos);
		}
		
		tm_update(etm->tsetlin_machine, Xi, 1);

		for (int patch = 0; patch < etm->number_of_patches; ++patch) {
			Xi[patch*etm->number_of_ta_chunks + feature_chunk] &= ~(1 << feature_chunk_pos);
			Xi[patch*etm->number_of_ta_chunks + inverted_feature_chunk] |= (1 << inverted_feature_chunk_pos);
		}
	} else {
		unsigned int negative_target_class = (unsigned int)etm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
		while (negative_target_class == target_class) {
			negative_target_class = (unsigned int)etm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
		}

		unsigned int feature_chunk = negative_target_class / 32;
		unsigned int feature_chunk_pos = negative_target_class % 32;

		unsigned int inverted_feature_chunk = (negative_target_class + etm->tsetlin_machine->number_of_features/2) / 32;
		unsigned int inverted_feature_chunk_pos = (negative_target_class + etm->tsetlin_machine->number_of_features/2) % 32;

		for (int patch = 0; patch < etm->number_of_patches; ++patch) {
			Xi[patch*etm->number_of_ta_chunks + feature_chunk] |= (1 << feature_chunk_pos);
			Xi[patch*etm->number_of_ta_chunks + inverted_feature_chunk] &= ~(1 << inverted_feature_chunk_pos);
		}

		tm_update(etm->tsetlin_machine, Xi, 0);

		for (int patch = 0; patch < etm->number_of_patches; ++patch) {
			Xi[patch*etm->number_of_ta_chunks + feature_chunk] &= ~(1 << feature_chunk_pos);
			Xi[patch*etm->number_of_ta_chunks + inverted_feature_chunk] |= (1 << inverted_feature_chunk_pos);
		}
	}
}

/**********************************************/
/*** Batch Mode Training of Tsetlin Machine ***/
/**********************************************/

void etm_fit(struct EmbeddingTsetlinMachine *etm, unsigned int *X, int *y, int number_of_examples, int epochs)
{
	unsigned int step_size = etm->number_of_patches * etm->number_of_ta_chunks;

	for (int epoch = 0; epoch < epochs; epoch++) {
		// Add shuffling here...
		unsigned int pos = 0;
		for (int i = 0; i < number_of_examples; i++) {
			etm_update(etm, &X[pos], y[i]);
			pos += step_size;
		}
	}
}

int etm_ta_state(struct EmbeddingTsetlinMachine *etm, int clause, int ta)
{
	return tm_ta_state(etm->tsetlin_machine, clause, ta);
}

int etm_ta_action(struct EmbeddingTsetlinMachine *etm, int clause, int ta)
{
	return tm_ta_action(etm->tsetlin_machine, clause, ta);
}

void etm_clause_configuration(struct EmbeddingTsetlinMachine *etm, int clause, unsigned int *clause_configuration)
{
	for (int k = 0; k < etm->tsetlin_machine->number_of_features; ++k) {
		clause_configuration[k] = tm_ta_action(etm->tsetlin_machine, clause, k);
	}

	return;
}

/*****************************************************/
/*** Storing and Loading of Tsetlin Machine State ****/
/*****************************************************/

void etm_get_state(struct EmbeddingTsetlinMachine *etm, unsigned int *clause_weights, unsigned int *ta_state)
{
	tm_get_ta_state(etm->tsetlin_machine, ta_state);
	tm_get_clause_weights(etm->tsetlin_machine, clause_weights);

	return;
}

void etm_set_state(struct EmbeddingTsetlinMachine *etm, unsigned int *clause_weights, unsigned int *ta_state)
{
	tm_set_ta_state(etm->tsetlin_machine, ta_state);
	tm_set_clause_weights(etm->tsetlin_machine, clause_weights);

	return;
}

/******************************************************************************/
/*** Clause Based Transformation of Input Examples for Multi-layer Learning ***/
/******************************************************************************/

void etm_transform(struct EmbeddingTsetlinMachine *etm, unsigned int *X,  unsigned int *X_transformed, int invert, int number_of_examples)
{
	unsigned int step_size = etm->number_of_patches * etm->number_of_ta_chunks;

	unsigned long pos = 0;
	unsigned long transformed_feature = 0;
	for (int l = 0; l < number_of_examples; l++) {
		unsigned int *Xi = &X[pos];

		for (int patch = 0; patch < etm->number_of_patches; ++patch) {
			for (int i = 0; i < etm->number_of_classes; i++) {	
				unsigned int feature_chunk = i / 32;
				unsigned int feature_chunk_pos = i % 32;

				Xi[patch*etm->number_of_ta_chunks + feature_chunk] |= (1 << feature_chunk_pos);
			}
		}

		tm_score(etm->tsetlin_machine, Xi);

		for (int patch = 0; patch < etm->number_of_patches; ++patch) {
			for (int i = 0; i < etm->number_of_classes; i++) {	
				unsigned int feature_chunk = i / 32;
				unsigned int feature_chunk_pos = i % 32;

				Xi[patch*etm->number_of_ta_chunks + feature_chunk] &= ~(1 << feature_chunk_pos);
			}
		}

		for (int j = 0; j < etm->tsetlin_machine->number_of_clauses; ++j) {
			int clause_chunk = j / 32;
			int clause_pos = j % 32;

			int clause_output = (etm->tsetlin_machine->clause_output[clause_chunk] & (1 << clause_pos)) > 0;

			if (clause_output && !invert) {
				X_transformed[transformed_feature] = 1;
			} else if (!clause_output && invert) {
				X_transformed[transformed_feature] = 1;
			} else {
				X_transformed[transformed_feature] = 0;
			}

			transformed_feature++;
		} 
		
		pos += step_size;
	}
	
	return;
}




