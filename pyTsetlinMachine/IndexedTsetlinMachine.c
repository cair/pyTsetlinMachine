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

#include "IndexedTsetlinMachine.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Creates index structure
struct IndexedTsetlinMachine *CreateIndexedTsetlinMachine(struct MultiClassTsetlinMachine *mc_tm)
{
	struct IndexedTsetlinMachine *itm = (void *)malloc(sizeof(struct IndexedTsetlinMachine));
	itm->mc_tm = mc_tm;

	itm->clause_state = (void *)malloc(sizeof(unsigned int) * itm->mc_tm->number_of_classes * itm->mc_tm->tsetlin_machines[0]->number_of_clauses * itm->mc_tm->tsetlin_machines[0]->number_of_features);

	itm->baseline_class_sum = (void *)malloc(sizeof(int) * itm->mc_tm->number_of_classes);

	itm->class_feature_list = (void *)malloc(sizeof(int) * itm->mc_tm->number_of_classes * (itm->mc_tm->tsetlin_machines[0]->number_of_clauses + 1) * itm->mc_tm->tsetlin_machines[0]->number_of_features * 2);
	itm->class_feature_pos = (void *)malloc(sizeof(int) * itm->mc_tm->number_of_classes * itm->mc_tm->tsetlin_machines[0]->number_of_clauses * itm->mc_tm->tsetlin_machines[0]->number_of_features);

	return itm;
}

void itm_destroy(struct IndexedTsetlinMachine *itm)
{
	free(itm->clause_state);
	free(itm->baseline_class_sum);
	free(itm->class_feature_list);
	free(itm->class_feature_pos);
	free(itm);
}

void itm_add_clause(struct IndexedTsetlinMachine *itm, int class, int clause, int feature)
{	
	int list_start = class*itm->mc_tm->tsetlin_machines[0]->number_of_features*(itm->mc_tm->tsetlin_machines[0]->number_of_clauses+1) + feature*(itm->mc_tm->tsetlin_machines[0]->number_of_clauses+1);
	int list_size = itm->class_feature_list[list_start];

	int list_pos = class*itm->mc_tm->tsetlin_machines[0]->number_of_clauses*itm->mc_tm->tsetlin_machines[0]->number_of_features + clause*itm->mc_tm->tsetlin_machines[0]->number_of_features + feature;
	itm->class_feature_list[list_start + list_size] = clause;
	itm->class_feature_pos[list_pos] = list_size;
	itm->class_feature_list[list_start]++;
}

void itm_delete_clause(struct IndexedTsetlinMachine *itm, int class, int clause, int feature)
{
	int list_pos = class*itm->mc_tm->tsetlin_machines[0]->number_of_clauses*itm->mc_tm->tsetlin_machines[0]->number_of_features + clause*itm->mc_tm->tsetlin_machines[0]->number_of_features + feature;

	int list_start = class*itm->mc_tm->tsetlin_machines[0]->number_of_features*(itm->mc_tm->tsetlin_machines[0]->number_of_clauses+1) + feature*(itm->mc_tm->tsetlin_machines[0]->number_of_clauses+1);
	int list_size = itm->class_feature_list[list_start];

	if (list_start + itm->class_feature_pos[list_pos] != list_start + list_size - 1) {
		itm->class_feature_list[list_start + itm->class_feature_pos[list_pos]] = itm->class_feature_list[list_start + list_size - 1];
		int list_pos_last = class*itm->mc_tm->tsetlin_machines[0]->number_of_clauses*itm->mc_tm->tsetlin_machines[0]->number_of_features + itm->class_feature_list[list_start + list_size - 1]*itm->mc_tm->tsetlin_machines[0]->number_of_features + feature;
		itm->class_feature_pos[list_pos_last] = itm->class_feature_pos[list_pos];
	}

	itm->class_feature_list[list_start]--;
	itm->class_feature_pos[list_pos] = -1;
}

void itm_initialize(struct IndexedTsetlinMachine *itm)
{
	int pos = 0;
	for (int i = 0; i < itm->mc_tm->number_of_classes; ++i) {
		for (int k = 0; k < itm->mc_tm->tsetlin_machines[0]->number_of_features; ++k) {
			itm->class_feature_list[pos] = 1; // Number of elements in list + 1
			pos += itm->mc_tm->tsetlin_machines[0]->number_of_clauses + 1;
		}
	}

	pos = 0;
	for (int i = 0; i < itm->mc_tm->number_of_classes; ++i) {
		for (int k = 0; k < itm->mc_tm->tsetlin_machines[0]->number_of_features; ++k) {
			for (int j = 0; j < itm->mc_tm->tsetlin_machines[0]->number_of_clauses; ++j) {
				itm->class_feature_pos[pos] = -1;
				pos++;
			}
		}
	}

	for (int k = 0; k < itm->mc_tm->tsetlin_machines[0]->number_of_features; ++k) {
		for (int i = 0; i < itm->mc_tm->number_of_classes; ++i) {
			for (int j = 0; j < itm->mc_tm->tsetlin_machines[i]->number_of_clauses; ++j) {
				if (mc_tm_ta_action(itm->mc_tm, i, j, k) == 1) {
					itm_add_clause(itm, i, j, k);
				}
			}
		}
	}
}

int itm_sum_up_clause_votes(struct IndexedTsetlinMachine *itm, int class, unsigned int *Xi)
{

	for (int j = 0; j < itm->mc_tm->tsetlin_machines[0]->number_of_clause_chunks; ++j) {
		itm->mc_tm->tsetlin_machines[class]->clause_output[j] = ~0;
	}

	int class_sum = 0;
	for (int k = 0; k < itm->mc_tm->tsetlin_machines[0]->number_of_features; ++k) {
		int k_chunk = k / 32;
		int k_pos = k % 32;

		if (!(Xi[k_chunk] & (1 << k_pos))) {
			int list_start = class*itm->mc_tm->tsetlin_machines[0]->number_of_features*(itm->mc_tm->tsetlin_machines[0]->number_of_clauses+1) + k*(itm->mc_tm->tsetlin_machines[0]->number_of_clauses+1);
			int list_size = itm->class_feature_list[list_start];

			for (int j = 1; j < list_size; j++) {
				int clause = itm->class_feature_list[list_start + j];
				int clause_chunk = clause / 32;
				int clause_pos = clause % 32;

				class_sum -= itm->mc_tm->tsetlin_machines[class]->clause_weights[clause]*(clause % 2 == 0)*((itm->mc_tm->tsetlin_machines[class]->clause_output[clause_chunk] & (1 << clause_pos)) > 0);	
				class_sum += itm->mc_tm->tsetlin_machines[class]->clause_weights[clause]*(clause % 2 != 0)*((itm->mc_tm->tsetlin_machines[class]->clause_output[clause_chunk] & (1 << clause_pos)) > 0);
				
				itm->mc_tm->tsetlin_machines[class]->clause_output[clause_chunk] &= ~(1 << clause_pos);
			}
		}
	}
	return class_sum;
}

void itm_predict(struct IndexedTsetlinMachine *itm, unsigned int *X, int *y, int number_of_examples)
{
	unsigned int step_size = itm->mc_tm->number_of_patches * itm->mc_tm->number_of_ta_chunks;

	// Initializes feature-clause map
	itm_initialize(itm);

	for (int i = 0; i < itm->mc_tm->number_of_classes; ++i) {
		itm->baseline_class_sum[i] = 0;

		for (int j = 0; j < itm->mc_tm->tsetlin_machines[0]->number_of_clauses; ++j) {
			int all_exclude = 1;
			for (int k = 0; k < itm->mc_tm->tsetlin_machines[0]->number_of_features; ++k) {
				if (mc_tm_ta_action(itm->mc_tm, i, j, k) == 1) {
					all_exclude = 0;
					break;
				}
			}

			if (!all_exclude) {
				itm->baseline_class_sum[i] += itm->mc_tm->tsetlin_machines[i]->clause_weights[j]*(1 - 2*(j % 2));
			}
		}
	}

	unsigned int pos = 0;
	for (int l = 0; l < number_of_examples; l++) {
		int max_class_sum = -itm->mc_tm->tsetlin_machines[0]->T-1;
		int max_class = -1;
		for (int i = 0; i < itm->mc_tm->number_of_classes; ++i) {	
			int class_sum = itm->baseline_class_sum[i] + itm_sum_up_clause_votes(itm, i, &X[pos]);
			class_sum = (class_sum > (itm->mc_tm->tsetlin_machines[0]->T)) ? (itm->mc_tm->tsetlin_machines[0]->T) : class_sum;
			class_sum = (class_sum < -(itm->mc_tm->tsetlin_machines[0]->T)) ? -(itm->mc_tm->tsetlin_machines[0]->T) : class_sum;

			if (max_class_sum < class_sum) {
				max_class_sum = class_sum;
				max_class = i;
			}
		}

		y[l] = max_class;

		pos += step_size;
	}
}

/**********************************************/
/*** Batch Mode Training of Tsetlin Machine ***/
/**********************************************/

void itm_fit(struct IndexedTsetlinMachine *itm, unsigned int *X, int y[], int number_of_examples, int epochs)
{
	// Initializes feature-clause map
	itm_initialize(itm);

	// Initializes clause state copy
	int pos = 0;
	for (int i = 0; i < itm->mc_tm->number_of_classes; ++i) {
		struct TsetlinMachine *tm = itm->mc_tm->tsetlin_machines[i];
		for (int j = 0; j < tm->number_of_clauses; ++j) {
			for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
				unsigned int current = tm->ta_state[j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1];
				itm->clause_state[pos] = current;
				pos++;
			}
		}
	}

	// Initialize the clause patch mapping, assuming one patch per input.
	for (int i = 0; i < itm->mc_tm->number_of_classes; ++i) {
		struct TsetlinMachine *tm = itm->mc_tm->tsetlin_machines[i];
		for (int j = 0; j < tm->number_of_clauses; ++j) {
			tm->clause_patch[j] = 0;
		}
	}

	unsigned int step_size = itm->mc_tm->number_of_patches * itm->mc_tm->number_of_ta_chunks;
	for (int epoch = 0; epoch < epochs; epoch++) {
		// Add shuffling here...
		unsigned int pos = 0;
		for (int l = 0; l < number_of_examples; l++) {
			itm_update(itm, &X[pos], y[l], 1);

			if (itm->mc_tm->number_of_classes > 1) {
				// Randomly pick one of the other classes, for pairwise learning of class output 
				unsigned int negative_target_class = (unsigned int)itm->mc_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
				while (negative_target_class == y[l]) {
					negative_target_class = (unsigned int)itm->mc_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
				}
				itm_update(itm, &X[pos], negative_target_class, 0);
			}
			
			pos += step_size;
		}
	}
}

/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void itm_update(struct IndexedTsetlinMachine *itm, unsigned int *Xi, int class, int target)
{
	struct TsetlinMachine *tm = itm->mc_tm->tsetlin_machines[class];

	int class_sum = 0;
	for (int j = 0; j < itm->mc_tm->tsetlin_machines[0]->number_of_clauses; ++j) {
		class_sum += itm->mc_tm->tsetlin_machines[class]->clause_weights[j]*(1 - 2*(j % 2));
	}
	class_sum += itm_sum_up_clause_votes(itm, class, Xi);

	class_sum = (class_sum > (itm->mc_tm->tsetlin_machines[0]->T)) ? (itm->mc_tm->tsetlin_machines[0]->T) : class_sum;
	class_sum = (class_sum < -(itm->mc_tm->tsetlin_machines[0]->T)) ? -(itm->mc_tm->tsetlin_machines[0]->T) : class_sum;

	tm_update_clauses(tm, Xi, class_sum, target);

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

		// Identifies clauses that have received feedback
		if (tm->feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos)) {
			for (int k = 0; k < tm->number_of_ta_chunks; ++k) {

				// Check whether actions have changed and update class-feature look-up tables accordingly
				unsigned int previous = itm->clause_state[class*tm->number_of_clauses*tm->number_of_ta_chunks + j*tm->number_of_ta_chunks + k];
				unsigned int current = tm->ta_state[j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1];
				if (previous != current) {
					for (int b = 0; b < 32; ++b) {
						if ((previous & (1 << b)) > 0 && (current & (1 << b)) == 0) {
							int feature = k*32 + b;
							if (feature < itm->mc_tm->tsetlin_machines[0]->number_of_features) {
								itm_delete_clause(itm, class, j, feature);
							}
						}

						if ((previous & (1 << b)) == 0 && (current & (1 << b)) > 0) {
							int feature = k*32 + b;
							if (feature < itm->mc_tm->tsetlin_machines[0]->number_of_features) {
								itm_add_clause(itm, class, j, feature);
							}
						}
					}
					itm->clause_state[class*tm->number_of_clauses*tm->number_of_ta_chunks + j*tm->number_of_ta_chunks + k] = current;
				}
			}
		}
	}
}

/******************************************************************************/
/*** Clause Based Transformation of Input Examples for Multi-layer Learning ***/
/******************************************************************************/

void itm_transform(struct IndexedTsetlinMachine *itm, unsigned int *X,  unsigned int *X_transformed, int invert, int number_of_examples)
{
	// Initializes feature-clause map
	itm_initialize(itm);

	unsigned int step_size = itm->mc_tm->number_of_patches * itm->mc_tm->number_of_ta_chunks;

	unsigned long pos = 0;
	unsigned long transformed_feature = 0;
	for (int l = 0; l < number_of_examples; l++) {
		for (int i = 0; i < itm->mc_tm->number_of_classes; i++) {	
			itm_sum_up_clause_votes(itm, i, &X[pos]);

			for (int j = 0; j < itm->mc_tm->tsetlin_machines[i]->number_of_clauses; ++j) {
				int clause_chunk = j / 32;
				int clause_pos = j % 32;

				int clause_output = (itm->mc_tm->tsetlin_machines[i]->clause_output[clause_chunk] & (1 << clause_pos)) > 0;
	
				if (clause_output && !invert) {
					X_transformed[transformed_feature] = 1;
				} else if (!clause_output && invert) {
					X_transformed[transformed_feature] = 1;
				} else {
					X_transformed[transformed_feature] = 0;
				}

				transformed_feature++;
			} 
		}
		pos += step_size;
	}
	
	return;
}
