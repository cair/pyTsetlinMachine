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

#define PREDICT 1
#define UPDATE 0

struct TsetlinMachine { 
	int number_of_clauses;

	int number_of_features;

	int number_of_clause_chunks;

	unsigned int *ta_state;
	unsigned int *clause_output;
	unsigned int *feedback_to_la;
	int *feedback_to_clauses;
	unsigned int *clause_patch;

	int *output_one_patches;

	int number_of_patches;
	int number_of_ta_chunks;
	int number_of_state_bits;

	int T;

	double s;

	unsigned int filter;

	int boost_true_positive_feedback;
};

struct TsetlinMachine *CreateTsetlinMachine(int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, int boost_true_positive_feedback);

void tm_initialize(struct TsetlinMachine *tm);

void tm_destroy(struct TsetlinMachine *tm);

void tm_update(struct TsetlinMachine *tm, unsigned int *Xi, int target);

int tm_score(struct TsetlinMachine *tm, unsigned int *Xi);

int tm_ta_state(struct TsetlinMachine *tm, int clause, int ta);

int tm_ta_action(struct TsetlinMachine *tm, int clause, int ta);

void tm_update_regression(struct TsetlinMachine *tm, unsigned int *Xi, int target);

void tm_fit_regression(struct TsetlinMachine *tm, unsigned int *X, int *y, int number_of_examples, int epochs);

int tm_score_regression(struct TsetlinMachine *tm, unsigned int *Xi);

void tm_predict_regression(struct TsetlinMachine *tm, unsigned int *X, int *y, int number_of_examples);

void tm_get_state(struct TsetlinMachine *tm, unsigned int *ta_state);

void tm_set_state(struct TsetlinMachine *tm, unsigned int *ta_state);
