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

#include "MultiClassConvolutionalTsetlinMachine.h"

struct IndexedTsetlinMachine { 
	struct MultiClassTsetlinMachine *mc_tm;

	unsigned int *clause_state; // Copy of current clauses, initalized at start of learning

	int *baseline_class_sum; // Sum of votes per class for baseline X

	int *class_feature_list; // (class, feature) look-up table that contains a list of clauses per clause-feature pair
	int *class_feature_pos;  // (class, clause, feature) look-up table that contains the position of each clause in every list
};

struct IndexedTsetlinMachine *CreateIndexedTsetlinMachine(struct MultiClassTsetlinMachine *mc_tm);

void itm_initialize(struct IndexedTsetlinMachine *itm);

void itm_fit(struct IndexedTsetlinMachine *itm, unsigned int *X, int *y, int number_of_examples, int epochs);

void itm_update(struct IndexedTsetlinMachine *itm, unsigned int *Xi, int class, int target);

void itm_predict(struct IndexedTsetlinMachine *itm, unsigned int *X, int *y, int number_of_examples);


