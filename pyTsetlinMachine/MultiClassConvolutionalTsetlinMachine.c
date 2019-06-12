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

#include <stdio.h>
#include <stdlib.h>

#include "MultiClassConvolutionalTsetlinMachine.h"

/**************************************/
/*** The Convolutional Tsetlin Machine ***/
/**************************************/

/*** Initialize Tsetlin Machine ***/
struct MultiClassTsetlinMachine *CreateMultiClassTsetlinMachine(int number_of_classes, int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, int boost_true_positive_feedback)
{

	struct MultiClassTsetlinMachine *mc_tm = NULL;

	mc_tm = (void *)malloc(sizeof(struct MultiClassTsetlinMachine));

	mc_tm->number_of_classes = number_of_classes;
	mc_tm->tsetlin_machines = (void *)malloc(sizeof(struct TsetlinMachine *)* number_of_classes);
	for (int i = 0; i < number_of_classes; i++) {
		mc_tm->tsetlin_machines[i] = CreateTsetlinMachine(number_of_clauses, number_of_features, number_of_patches, number_of_ta_chunks, number_of_state_bits, T, s, boost_true_positive_feedback);
	}
	
	mc_tm->number_of_patches = number_of_patches;

	mc_tm->number_of_ta_chunks = number_of_ta_chunks;

	mc_tm->number_of_state_bits = number_of_state_bits;

	return mc_tm;
}

void mc_tm_initialize(struct MultiClassTsetlinMachine *mc_tm)
{
	for (int i = 0; i < mc_tm->number_of_classes; i++) {
		tm_initialize(mc_tm->tsetlin_machines[i]);
	}
}

void mc_tm_destroy(struct MultiClassTsetlinMachine *mc_tm)
{
	for (int i = 0; i < mc_tm->number_of_classes; i++) {
		tm_destroy(mc_tm->tsetlin_machines[i]);

		free(mc_tm->tsetlin_machines[i]);
	}
	free(mc_tm->tsetlin_machines);
}

/********************************************/
/*** Evaluate the Trained Tsetlin Machine ***/
/********************************************/

float mc_tm_evaluate(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int number_of_examples)
{
	int errors;
	int max_class;
	int max_class_sum;

	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

	unsigned int pos = 0;

	errors = 0;
	for (int l = 0; l < number_of_examples; l++) {
		/******************************************/
		/*** Identify Class with Largest Output ***/
		/******************************************/

		max_class_sum = tm_score(mc_tm->tsetlin_machines[0], &X[pos]);
		max_class = 0;
		for (int i = 1; i < mc_tm->number_of_classes; i++) {	
			int class_sum = tm_score(mc_tm->tsetlin_machines[i], &X[pos]);
			if (max_class_sum < class_sum) {
				max_class_sum = class_sum;
				max_class = i;
			}
		}

		if (max_class != y[l]) {
			errors += 1;
		}
		pos += step_size;
	}
	
	return 1.0 - 1.0 * errors / number_of_examples;
}

/*********************************/
/*** Predict class of input Xi ***/
/*********************************/

int mc_tm_predict(struct MultiClassTsetlinMachine *mc_tm, unsigned int *Xi)
{
	int max_class;
	int max_class_sum;
	
	/******************************************/
	/*** Identify Class with Largest Output ***/
	/******************************************/

	max_class_sum = tm_score(mc_tm->tsetlin_machines[0], Xi);
	max_class = 0;
	for (int i = 1; i < mc_tm->number_of_classes; i++) {	
		int class_sum = tm_score(mc_tm->tsetlin_machines[i], Xi);
		if (max_class_sum < class_sum) {
			max_class_sum = class_sum;
			max_class = i;
		}
	}	
	
	return max_class;
}


/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void mc_tm_update(struct MultiClassTsetlinMachine *mc_tm, unsigned int *Xi, int target_class)
{
	tm_update(mc_tm->tsetlin_machines[target_class], Xi, 1);

	// Randomly pick one of the other classes, for pairwise learning of class output 
	unsigned int negative_target_class = (unsigned int)mc_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	while (negative_target_class == target_class) {
		negative_target_class = (unsigned int)mc_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	}
	tm_update(mc_tm->tsetlin_machines[negative_target_class], Xi, 0);
}

/**********************************************/
/*** Batch Mode Training of Tsetlin Machine ***/
/**********************************************/

void mc_tm_fit(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int number_of_examples, int epochs)
{
	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

	for (int epoch = 0; epoch < epochs; epoch++) {
		// Add shuffling here...
		unsigned int pos = 0;
		for (int i = 0; i < number_of_examples; i++) {
			mc_tm_update(mc_tm, &X[pos], y[i]);
			pos += step_size;
		}
	}
}

