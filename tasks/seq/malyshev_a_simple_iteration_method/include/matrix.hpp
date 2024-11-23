#pragma once

#include <cinttypes>
#include <cmath>
#include <ctime>
#include <limits>
#include <random>
#include <vector>

namespace malyshev_a_simple_iteration_method_seq {

double determinant(const std::vector<std::vector<double>>& matrix);
double determinant(const std::vector<double>& matrix, uint32_t n);
int rank(const std::vector<std::vector<double>>& matrix);
void getRandomData(uint32_t n, std::vector<double>& A, std::vector<double>& B);

}  // namespace malyshev_a_simple_iteration_method_seq