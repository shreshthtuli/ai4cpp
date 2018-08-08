/*
Author: Shreshth Tuli
Date:   30/07/2018

This program functions as the "brain" of a reinforcement learning
agent whose goal is to provide the optimal values of priorities
of different processor groups for the Cache bandwidth allocation.
*/

#include <iostream>
#include <random>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <list>
#include <limits>
#include <cstdlib>
#include <ctime>

using namespace std;

class RL
{
	// Number of state parameters
	static const int num_params = 2;

	// Value range 1 to n
	static const int range = 4;

	// State intialized with all values = 0
	public : int state[num_params] = { 0 };

	// Number of states and actions
	static const int num_states = int(pow(range, num_params));
	static const int num_actions = 1 + (2 * num_params);

	// Initialize Q
	float Q[num_states][num_actions];

	//Computational parameters
	float gamma_q = 0.4;      //look-ahead weight
	float alpha = 0.90;        //"Forgetfulness" weight.  The closer this is to 1 the more weight is given to recent samples.
							   //A high value is kept because of a highly dynamic situation, we cannot keep it very high as then the system might not converge

	//Parameters for getAction()
	public : float epsilon = 1.0;             //epsilon is the probability of choosing an action randomly.  1-epsilon is the probability of choosing the optimal action

	// State Index
	int s = 0;
	int sPrime = 0;

	// Performance variables
	float distanceNew = 0.0;
	float distanceOld = 0.0;
	float deltaDistance = 0.0;

	// Main loop
	float r = 0.0;
	float lookAheadValue = 0.0;
	float sample = 0.0;
	int a = 0;
	list<int>::iterator it;
	int readDelay = 10;
	float explorationMinutes = 0.1;
	float explorationConst = (explorationMinutes*60.0) / ((float(readDelay)) / 1000.0);  
	//this is the approximate exploration time in units of number of times through the loop
	int t = 0;

	// State parameters in this->state array are 1 less of actual value

	public: void init() {

		cout << "RL Module intialized with " << num_params << " parameters and range = " << range << endl;

		cout << "Number of states = " << num_states << endl;
		cout << "Number of actions = " << num_actions << endl;

		for (int i = 0; i < num_params; i++) {
			s += (state[i] * int(pow(range, i)));
		}

		sPrime = s;

	};

	// Returns action as 0, 1, 2, ... = NONE, theta1++, theta1--, theta2++, ...
	private: int getAction() {
		int action;
		float valMax = -100000000.0;
		float val;
		int aMax;
		float randVal;
		int allowedActions[num_actions] = { -1 };
		allowedActions[0] = 1;

		bool randomActionFound = false;

		// Find the optimal action, and exclude that take you outside the allowed state space
		val = Q[s][0];
		if (val > valMax) {
			valMax = val;
			aMax = 0;
		}

		for (int i = 0; i < num_params; i++) {

			if (state[i] + 1 < range) {
				allowedActions[2 * i + 1] = 1;
				val = Q[s][2 * i + 1];
				if (val > valMax) {
					valMax = val;
					aMax = 2 * i + 1;
				}
			}

			if (state[i] > 0) {
				allowedActions[2 * i + 2] = 1;
				val = Q[s][2 * i + 2];
				if (val > valMax) {
					valMax = val;
					aMax = 2 * i + 2;
				}
			}
		}

		// Implement epsilon greedy alogorithm
		randVal = float(rand() % 101);
		if(randVal < (1.0 - epsilon)*100.0){
			action = aMax;
		}
		else {
			while (!randomActionFound) {
				action = int(rand() % num_actions);
				if (allowedActions[action] == 1) {
					randomActionFound = true;
				}
			}
		}

		return(action);
	};

	//Given a and the global(s) find the next state.  Also keep track of the individual joint indexes s1 and s2.
	private: void setSPrime(int action) {
		if (action == 0) {
			// NONE
			sPrime = s;
		}
		else {
			int power = (action - 1) / 2;
			int absolute = int(pow(range, power));
			if (action % 2 == 1) {
				sPrime = s + absolute;
				state[power] = state[power] + 1;
			}
			else {
				sPrime = s - absolute;
				state[power] = state[power] - 1;
			}
		}
	};

	// Distance here means performance, goal is to maximize distance
	// by tuning the hyper parameters inside the performance function 
	// Q learning algorithm 
	
	//Get the reward using the increase in performance since the last call
	float getDeltaDistance(float ipc) {
		distanceNew = ipc;

		cout << "New performance : " << ipc << endl;
		deltaDistance = distanceNew - distanceOld;

		distanceOld = distanceNew;
		return deltaDistance;
	};

	//Get max over a' of Q(s',a'), but be careful not to look at actions which take the agent outside of the allowed state space
	float getLookAhead() {
		float valMax = -100000.0;
		float val;

		val = Q[sPrime][0];
		if (val > valMax) {
			valMax = val;
		}

		for (int i = 0; i < num_params; i++) {

			if (state[i] + 1 < range) {
				val = Q[sPrime][2 * i + 1];
				if (val > valMax) {
					valMax = val;
				}
			}

			if (state[i] > 0) {
				val = Q[sPrime][2 * i + 2];
				if (val > valMax) {
					valMax = val;
				}
			}
		}

		return valMax;
	}

	void initializeQ() {
		for (int i = 0; i < num_states; i++) {
			for (int j = 0; j < num_actions; j++) {
				Q[i][j] = 10.0;               //Initialize to a positive number to represent optimism over all state-actions
			}
		}
	};

	//print Q
	void printQ() {
		cout << "Q is: " << endl;
		for (int i = 0; i < num_states; i++) {
			cout << i << "\t\t\t";
			for (int j = 0; j < num_actions; j++) {
				cout << Q[i][j] << "\t \t \t";
			};
			cout << endl;
		};
	};

	public : int iterate(float ipc) {
		t++;
		cout << "Iteration number : " << t << endl;
		epsilon = exp(-float(t) / explorationConst);
		cout << "Epsilon: " << epsilon << endl;
		a = getAction();
		setSPrime(a);
		cout << "Action : " << a << endl;
		r = getDeltaDistance(ipc);
		lookAheadValue = getLookAhead();
		sample = r + gamma_q * lookAheadValue;
		Q[s][a] = Q[s][a] + alpha * (sample - Q[s][a]);
		s = sPrime;

		cout << "State : ";
		for (int i = 0; i < num_params; i++) {
			cout << state[i] << ", ";
		}
		cout << endl << endl;

		if (t == 2) {
			initializeQ();
		}

		//printQ();
	}


};

int main() {
	RL test;
	test.init();

	int input;
	while (test.epsilon > 0.0003) {
		input = 100 - pow((test.state[0] - 1), 4) - pow((test.state[1] - 3), 4);
		test.iterate(input);
	}
}