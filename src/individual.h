/*
 * individual.h
 *
 *  Created on: Nov 1, 2016
 *      Author: thogan
 */

#ifndef INDIVIDUAL_H_
#define INDIVIDUAL_H_

struct coord2D
{
	long x;
	long y;
};

struct Individual {
	int id;
	curandState_t state;
	double fitness;
	coord2D chromosome[128];
};

#endif /* INDIVIDUAL_H_ */
