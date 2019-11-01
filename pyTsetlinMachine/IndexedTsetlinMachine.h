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


