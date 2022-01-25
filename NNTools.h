#pragma once

#include <bits/stdc++.h>
#include <cmath>

static double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

static double dsigmoid(double x) {
	return x * (1 - x);
}