// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/korotin_e_multidimentional_integrals_monte_carlo/include/ops_seq.hpp"

namespace korotin_e_multidimentional_integrals_monte_carlo_seq {

double test_func(const double *x, int x_size) {
  double res = 0.0;
  for (int i = 0; i < x_size; i++) {
    res += std::pow(x[i], 2);
  }
  return res;
}

double ref_integration(const std::vector<std::pair<double, double>> &borders) {
  double res = 0.0;
  for (size_t i = 0; i < borders.size(); i++) {
    double tmp = std::pow(borders[i].second, 3);
    tmp -= std::pow(borders[i].first, 3);
    tmp /= 3;
    for (size_t j = 0; j < borders.size(); j++) {
      if (j == i) continue;
      tmp *= borders[j].second - borders[j].first;
    }
    res += tmp;
  }
  return res;
}

}  // namespace korotin_e_multidimentional_integrals_monte_carlo_seq

TEST(korotin_e_multidimentional_integrals_monte_carlo_seq, monte_carlo_rng_borders) {
  std::vector<std::pair<double, double>> borders(3);
  std::vector<double> res(1, 0);
  std::vector<size_t> N(1, 500);
  std::vector<double (*)(const double *, int)> F(1, &korotin_e_multidimentional_integrals_monte_carlo_seq::test_func);

  double ref;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(-3.0, 3.0);
  for (int i = 0; i < 3; i++) {
    borders[i].first = distrib(gen);
    borders[i].second = distrib(gen);
  }

  ref = korotin_e_multidimentional_integrals_monte_carlo_seq::ref_integration(borders);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(F.data()));
  taskDataSeq->inputs_count.emplace_back(F.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(borders.data()));
  taskDataSeq->inputs_count.emplace_back(borders.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(N.data()));
  taskDataSeq->inputs_count.emplace_back(N.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  korotin_e_multidimentional_integrals_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  double err = testTaskSequential.possible_error();
  bool ans = (std::abs(res[0] - ref) < err);
  ASSERT_EQ(ans, true);
}

TEST(korotin_e_multidimentional_integrals_monte_carlo_seq, monte_carlo_pseudo_rng_func) {
  int dimentions = rand() % 5 + 1;
  std::vector<std::pair<double, double>> borders(dimentions);
  std::vector<double> res(1, 0);
  std::vector<size_t> N(1, 500);
  std::vector<double (*)(const double *, int)> F(1, &korotin_e_multidimentional_integrals_monte_carlo_seq::test_func);
  double ref;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(-3.0, 3.0);
  for (int i = 0; i < dimentions; i++) {
    borders[i].first = distrib(gen);
    borders[i].second = distrib(gen);
  }

  ref = korotin_e_multidimentional_integrals_monte_carlo_seq::ref_integration(borders);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(F.data()));
  taskDataSeq->inputs_count.emplace_back(F.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(borders.data()));
  taskDataSeq->inputs_count.emplace_back(borders.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(N.data()));
  taskDataSeq->inputs_count.emplace_back(N.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  korotin_e_multidimentional_integrals_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  double err = testTaskSequential.possible_error();
  bool ans = (std::abs(res[0] - ref) < err);
  ASSERT_EQ(ans, true);
}
