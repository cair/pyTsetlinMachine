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

#include "ClassInputConvolutionalTsetlinMachine.h"

/**************************************/
/*** The Convolutional Tsetlin Machine ***/
/**************************************/

/*** Initialize Tsetlin Machine ***/
struct ClassInputTsetlinMachine *CreateClassInputTsetlinMachine(int number_of_classes, int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range, int boost_true_positive_feedback, int weighted_clauses)
{
	struct ClassInputTsetlinMachine *ci_tm = (void *)malloc(sizeof(struct ClassInputTsetlinMachine));

	ci_tm->number_of_classes = number_of_classes;

	ci_tm->tsetlin_machine = CreateTsetlinMachine(number_of_clauses, number_of_features, number_of_patches, number_of_ta_chunks, number_of_state_bits, T, s, s_range, boost_true_positive_feedback, weighted_clauses);
	
	ci_tm->number_of_patches = number_of_patches;

	ci_tm->number_of_ta_chunks = number_of_ta_chunks;

	ci_tm->number_of_state_bits = number_of_state_bits;

	return ci_tm;
}

void ci_tm_initialize(struct ClassInputTsetlinMachine *ci_tm)
{
	tm_initialize(ci_tm->tsetlin_machine);
}

void ci_tm_destroy(struct ClassInputTsetlinMachine *ci_tm)
{
	tm_destroy(ci_tm->tsetlin_machine);

	free(ci_tm->tsetlin_machine);
}

/***********************************/
/*** Predict classes of inputs X ***/
/***********************************/

void ci_tm_predict(struct ClassInputTsetlinMachine *ci_tm, unsigned int *X, int *y, int number_of_examples)
{
	int max_class;
	int max_class_sum;

	unsigned int step_size = ci_tm->number_of_patches * ci_tm->number_of_ta_chunks;

	unsigned int pos = 0;

	for (int l = 0; l < number_of_examples; l++) {
		// Identify class with largest output
		max_class_sum = -ci_tm->tsetlin_machine->T;
		max_class = 0;
		for (int i = 0; i < ci_tm->number_of_classes; i++) {	
			unsigned int feature_chunk = i / 32;
			unsigned int feature_chunk_pos = i % 32;

			unsigned int inverted_feature_chunk = (i + ci_tm->tsetlin_machine->number_of_features/2) / 32;
			unsigned int inverted_feature_chunk_pos = (i + ci_tm->tsetlin_machine->number_of_features/2) % 32;

			unsigned int *Xi = &X[pos];

			for (int patch = 0; patch < ci_tm->number_of_patches; ++patch) {
				Xi[patch*ci_tm->number_of_ta_chunks + feature_chunk] |= (1 << feature_chunk_pos);
				Xi[patch*ci_tm->number_of_ta_chunks + inverted_feature_chunk] &= ~(1 << inverted_feature_chunk_pos);
			}

			int class_sum = tm_score(ci_tm->tsetlin_machine, Xi);

			for (int patch = 0; patch < ci_tm->number_of_patches; ++patch) {
				Xi[patch*ci_tm->number_of_ta_chunks + feature_chunk] &= ~(1 << feature_chunk_pos);
				Xi[patch*ci_tm->number_of_ta_chunks + inverted_feature_chunk] |= (1 << inverted_feature_chunk_pos);
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

void ci_tm_update_alt(struct ClassInputTsetlinMachine *ci_tm, unsigned int *Xi, int target_class)
{

	unsigned int training_class = (unsigned int)ci_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	
	unsigned int feature_chunk = training_class / 32;
	unsigned int feature_chunk_pos = training_class % 32;

	unsigned int inverted_feature_chunk = (training_class + ci_tm->tsetlin_machine->number_of_features/2) / 32;
	unsigned int inverted_feature_chunk_pos = (training_class + ci_tm->tsetlin_machine->number_of_features/2) % 32;

	for (int patch = 0; patch < ci_tm->number_of_patches; ++patch) {
		Xi[patch*ci_tm->number_of_ta_chunks + feature_chunk] |= (1 << feature_chunk_pos);
		Xi[patch*ci_tm->number_of_ta_chunks + inverted_feature_chunk] &= ~(1 << inverted_feature_chunk_pos);
	}

	if (training_class == target_class) {
		tm_update(ci_tm->tsetlin_machine, Xi, 1);
	} else {
		tm_update(ci_tm->tsetlin_machine, Xi, 0);
	}

	for (int patch = 0; patch < ci_tm->number_of_patches; ++patch) {
		Xi[patch*ci_tm->number_of_ta_chunks + feature_chunk] &= ~(1 << feature_chunk_pos);
		Xi[patch*ci_tm->number_of_ta_chunks + inverted_feature_chunk] |= (1 << inverted_feature_chunk_pos);
	}
}

void ci_tm_update(struct ClassInputTsetlinMachine *ci_tm, unsigned int *Xi, int target_class)
{
	if (1.0*rand()/((unsigned int)RAND_MAX) <= 0.5) {
		unsigned int feature_chunk = target_class / 32;
		unsigned int feature_chunk_pos = target_class % 32;

		unsigned int inverted_feature_chunk = (target_class + ci_tm->tsetlin_machine->number_of_features/2) / 32;
		unsigned int inverted_feature_chunk_pos = (target_class + ci_tm->tsetlin_machine->number_of_features/2) % 32;

		for (int patch = 0; patch < ci_tm->number_of_patches; ++patch) {
			Xi[patch*ci_tm->number_of_ta_chunks + feature_chunk] |= (1 << feature_chunk_pos);
			Xi[patch*ci_tm->number_of_ta_chunks + inverted_feature_chunk] &= ~(1 << inverted_feature_chunk_pos);
		}
		
		tm_update(ci_tm->tsetlin_machine, Xi, 1);

		for (int patch = 0; patch < ci_tm->number_of_patches; ++patch) {
			Xi[patch*ci_tm->number_of_ta_chunks + feature_chunk] &= ~(1 << feature_chunk_pos);
			Xi[patch*ci_tm->number_of_ta_chunks + inverted_feature_chunk] |= (1 << inverted_feature_chunk_pos);
		}
	} else {
		unsigned int negative_target_class = (unsigned int)ci_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
		while (negative_target_class == target_class) {
			negative_target_class = (unsigned int)ci_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
		}

		unsigned int feature_chunk = negative_target_class / 32;
		unsigned int feature_chunk_pos = negative_target_class % 32;

		unsigned int inverted_feature_chunk = (negative_target_class + ci_tm->tsetlin_machine->number_of_features/2) / 32;
		unsigned int inverted_feature_chunk_pos = (negative_target_class + ci_tm->tsetlin_machine->number_of_features/2) % 32;

		for (int patch = 0; patch < ci_tm->number_of_patches; ++patch) {
			Xi[patch*ci_tm->number_of_ta_chunks + feature_chunk] |= (1 << feature_chunk_pos);
			Xi[patch*ci_tm->number_of_ta_chunks + inverted_feature_chunk] &= ~(1 << inverted_feature_chunk_pos);
		}

		tm_update(ci_tm->tsetlin_machine, Xi, 0);

		for (int patch = 0; patch < ci_tm->number_of_patches; ++patch) {
			Xi[patch*ci_tm->number_of_ta_chunks + feature_chunk] &= ~(1 << feature_chunk_pos);
			Xi[patch*ci_tm->number_of_ta_chunks + inverted_feature_chunk] |= (1 << inverted_feature_chunk_pos);
		}
	}
}

/**********************************************/
/*** Batch Mode Training of Tsetlin Machine ***/
/**********************************************/

void ci_tm_fit(struct ClassInputTsetlinMachine *ci_tm, unsigned int *X, int *y, int number_of_examples, int epochs)
{
	unsigned int step_size = ci_tm->number_of_patches * ci_tm->number_of_ta_chunks;

	for (int epoch = 0; epoch < epochs; epoch++) {
		// Add shuffling here...
		unsigned int pos = 0;
		for (int i = 0; i < number_of_examples; i++) {
			ci_tm_update(ci_tm, &X[pos], y[i]);
			pos += step_size;
		}
	}
}

int ci_tm_ta_state(struct ClassInputTsetlinMachine *ci_tm, int clause, int ta)
{
	return tm_ta_state(ci_tm->tsetlin_machine, clause, ta);
}

int ci_tm_ta_action(struct ClassInputTsetlinMachine *ci_tm, int clause, int ta)
{
	return tm_ta_action(ci_tm->tsetlin_machine, clause, ta);
}

void ci_tm_clause_configuration(struct ClassInputTsetlinMachine *ci_tm, int clause, unsigned int *clause_configuration)
{
	for (int k = 0; k < ci_tm->tsetlin_machine->number_of_features; ++k) {
		clause_configuration[k] = tm_ta_action(ci_tm->tsetlin_machine, clause, k);
	}

	return;
}

/*****************************************************/
/*** Storing and Loading of Tsetlin Machine State ****/
/*****************************************************/

void ci_tm_get_state(struct ClassInputTsetlinMachine *ci_tm, unsigned int *clause_weights, unsigned int *ta_state)
{
	tm_get_ta_state(ci_tm->tsetlin_machine, ta_state);
	tm_get_clause_weights(ci_tm->tsetlin_machine, clause_weights);

	return;
}

void ci_tm_set_state(struct ClassInputTsetlinMachine *ci_tm, unsigned int *clause_weights, unsigned int *ta_state)
{
	tm_set_ta_state(ci_tm->tsetlin_machine, ta_state);
	tm_set_clause_weights(ci_tm->tsetlin_machine, clause_weights);

	return;
}

/******************************************************************************/
/*** Clause Based Transformation of Input Examples for Multi-layer Learning ***/
/******************************************************************************/

void ci_tm_transform(struct ClassInputTsetlinMachine *ci_tm, unsigned int *X,  unsigned int *X_transformed, int invert, int number_of_examples)
{
	unsigned int step_size = ci_tm->number_of_patches * ci_tm->number_of_ta_chunks;

	unsigned long pos = 0;
	unsigned long transformed_feature = 0;
	for (int l = 0; l < number_of_examples; l++) {
		unsigned int *Xi = &X[pos];

		for (int patch = 0; patch < ci_tm->number_of_patches; ++patch) {
			for (int i = 0; i < ci_tm->number_of_classes; i++) {	
				unsigned int feature_chunk = i / 32;
				unsigned int feature_chunk_pos = i % 32;

				Xi[patch*ci_tm->number_of_ta_chunks + feature_chunk] |= (1 << feature_chunk_pos);
			}
		}

		tm_score(ci_tm->tsetlin_machine, Xi);

		for (int patch = 0; patch < ci_tm->number_of_patches; ++patch) {
			for (int i = 0; i < ci_tm->number_of_classes; i++) {	
				unsigned int feature_chunk = i / 32;
				unsigned int feature_chunk_pos = i % 32;

				Xi[patch*ci_tm->number_of_ta_chunks + feature_chunk] &= ~(1 << feature_chunk_pos);
			}
		}

		for (int j = 0; j < ci_tm->tsetlin_machine->number_of_clauses; ++j) {
			int clause_chunk = j / 32;
			int clause_pos = j % 32;

			int clause_output = (ci_tm->tsetlin_machine->clause_output[clause_chunk] & (1 << clause_pos)) > 0;

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




