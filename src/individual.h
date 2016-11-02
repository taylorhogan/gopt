/*
 * individual.h
 *
 *  Created on: Nov 1, 2016
 *      Author: thogan
 */

#ifndef INDIVIDUAL_H_
#define INDIVIDUAL_H_

struct Individual
{
	int id;
	curandState_t state;
	double fitness;
	int chromosome[128];
};





#endif /* INDIVIDUAL_H_ */
