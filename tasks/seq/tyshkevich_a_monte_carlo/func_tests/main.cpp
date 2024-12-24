#include <gtest/gtest.h>

#include "seq/tyshkevich_a_monte_carlo/include/ops_seq.hpp"
#include "seq/tyshkevich_a_monte_carlo/include/test_include.hpp"

namespace tyshkevich_a_monte_carlo_seq {

void test(std::function<double(const std::vector<double>&)> function, double exp_res, int dimensions = 3,
          double precision = 100000, double left_bound = 0.0, double right_bound = 1.0) {
  double result = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dimensions));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&precision));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left_bound));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right_bound));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  MonteCarloSequential testMpiTaskParallel(taskDataPar, std::move(function));
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  EXPECT_NEAR(result, exp_res, 0.1);
}

}  // namespace tyshkevich_a_monte_carlo_seq

TEST(tyshkevich_a_monte_carlo_seq, function_sin_sum_dims_1_little_bounds) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_sin_sum, 4.498536054991164e-36, 1,
                                     0.000000001, 0.000000002);
}

TEST(tyshkevich_a_monte_carlo_seq, function_sin_sum_dims_2_little_bounds) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_sin_sum, 3.000190137524261e-27, 2,
                                     0.000000001, 0.000000002);
}

TEST(tyshkevich_a_monte_carlo_seq, function_sin_sum_dims_3_little_bounds) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_sin_sum, 4.500812680725791e-36, 3,
                                     0.000000001, 0.000000002);
}

TEST(tyshkevich_a_monte_carlo_seq, function_sin_sum_dims_4_little_bounds) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_sin_sum, 6.001340058062818e-45, 4,
                                     0.000000001, 0.000000002);
}

TEST(tyshkevich_a_monte_carlo_seq, function_sin_sum_dims_1) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_sin_sum, 0.459698, 1);
}

TEST(tyshkevich_a_monte_carlo_seq, function_cos_product_dims_1) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_cos_product, 0.841471, 1);
}

TEST(tyshkevich_a_monte_carlo_seq, function_gaussian_dims_1) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_gaussian, 0.746824, 1);
}

TEST(tyshkevich_a_monte_carlo_seq, function_paraboloid_dims_1) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_paraboloid, 0.333333, 1);
}

TEST(tyshkevich_a_monte_carlo_seq, function_exp_sum_dims_1) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_exp_sum, 1.718282, 1);
}

TEST(tyshkevich_a_monte_carlo_seq, function_abs_product_dims_1) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_abs_product, 0.500000, 1);
}

TEST(tyshkevich_a_monte_carlo_seq, function_log_sum_squares_dims_1) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_log_sum_squares, 0.263944, 1);
}

TEST(tyshkevich_a_monte_carlo_seq, function_sin_sum_dims_2) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_sin_sum, 0.919395, 2);
}

TEST(tyshkevich_a_monte_carlo_seq, function_cos_product_dims_2) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_cos_product, 0.708073, 2);
}

TEST(tyshkevich_a_monte_carlo_seq, function_gaussian_dims_2) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_gaussian, 0.557746, 2);
}

TEST(tyshkevich_a_monte_carlo_seq, function_paraboloid_dims_2) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_paraboloid, 0.666667, 2);
}

TEST(tyshkevich_a_monte_carlo_seq, function_exp_sum_dims_2) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_exp_sum, 3.436564, 2);
}

TEST(tyshkevich_a_monte_carlo_seq, function_abs_product_dims_2) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_abs_product, 0.250000, 2);
}

TEST(tyshkevich_a_monte_carlo_seq, function_log_sum_squares_dims_2) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_log_sum_squares, 0.478962, 2);
}

TEST(tyshkevich_a_monte_carlo_seq, function_sin_sum_dims_3) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_sin_sum, 1.379093, 3);
}

TEST(tyshkevich_a_monte_carlo_seq, function_cos_product_dims_3) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_cos_product, 0.595823, 3);
}

TEST(tyshkevich_a_monte_carlo_seq, function_gaussian_dims_3) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_gaussian, 0.416538, 3);
}

TEST(tyshkevich_a_monte_carlo_seq, function_paraboloid_dims_3) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_paraboloid, 1.000000, 3);
}

TEST(tyshkevich_a_monte_carlo_seq, function_exp_sum_dims_3) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_exp_sum, 5.154845, 3);
}

TEST(tyshkevich_a_monte_carlo_seq, function_abs_product_dims_3) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_abs_product, 0.125000, 3);
}

TEST(tyshkevich_a_monte_carlo_seq, function_log_sum_squares_dims_3) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_log_sum_squares, 0.659136, 3);
}

TEST(tyshkevich_a_monte_carlo_seq, function_sin_sum_dims_4) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_sin_sum, 1.838791, 4);
}

TEST(tyshkevich_a_monte_carlo_seq, function_cos_product_dims_4) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_cos_product, 0.501368, 4);
}

TEST(tyshkevich_a_monte_carlo_seq, function_gaussian_dims_4) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_gaussian, 0.311081, 4);
}

TEST(tyshkevich_a_monte_carlo_seq, function_paraboloid_dims_4) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_paraboloid, 1.333333, 4);
}

TEST(tyshkevich_a_monte_carlo_seq, function_exp_sum_dims_4) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_exp_sum, 6.873127, 4);
}

TEST(tyshkevich_a_monte_carlo_seq, function_abs_product_dims_4) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_abs_product, 0.062500, 4);
}

TEST(tyshkevich_a_monte_carlo_seq, function_log_sum_squares_dims_4) {
  tyshkevich_a_monte_carlo_seq::test(tyshkevich_a_monte_carlo_seq::function_log_sum_squares, 0.813557, 4);
}
