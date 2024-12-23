#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "seq/grudzin_k_monte_carlo/include/ops_seq.hpp"

namespace grudzin_k_montecarlo_seq {
std::vector<double> GenDimDistr(int dim, double left = -5.0, double right = 5.0) {
  std::mt19937 rnd;
  std::uniform_real_distribution<> dist(left, right);
  std::vector<double> tmp;
  for (int i = 0; i < dim; ++i) {
    double start = dist(rnd);
    double finish = dist(rnd);
    if (finish < start) std::swap(start, finish);
    tmp.emplace_back(start);
    tmp.emplace_back(finish);
  }
  return tmp;
}

double CalcEtalon(std::vector<double> &dim) {
  double ans = 1;
  for (size_t i = 0; i < dim.size(); i += 2) {
    ans *= dim[i + 1] - dim[i];
  }
  return ans;
}
}  // namespace grudzin_k_montecarlo_seq

TEST(grudzin_k_monte_carlo_seq, Test_1Dim) {
  const int dimensions = 1;
  int N = 10000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * x[0] + 1.0; }
  };
  std::vector<double> dim = grudzin_k_montecarlo_seq::GenDimDistr(dimensions);
  double etalon = grudzin_k_montecarlo_seq::CalcEtalon(dim);
  double result_seq = 0;
  std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  MC2_Data->inputs_count.emplace_back(dimensions);
  MC2_Data->inputs_count.emplace_back(1);
  MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
  MC2_Data->outputs_count.emplace_back(1);

  grudzin_k_montecarlo_seq::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
  ASSERT_EQ(MC2.validation(), true);
  MC2.pre_processing();
  MC2.run();
  MC2.post_processing();
  EXPECT_LE(abs(etalon - result_seq) / etalon, 1e-1);
}

TEST(grudzin_k_monte_carlo_seq, Test_2Dim) {
  const int dimensions = 2;
  int N = 10000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * (x[0] + x[1]) + 1.0; }
  };
  std::vector<double> dim = grudzin_k_montecarlo_seq::GenDimDistr(dimensions);
  double etalon = grudzin_k_montecarlo_seq::CalcEtalon(dim);
  double result_seq = 0;
  std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  MC2_Data->inputs_count.emplace_back(dimensions);
  MC2_Data->inputs_count.emplace_back(1);
  MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
  MC2_Data->outputs_count.emplace_back(1);

  grudzin_k_montecarlo_seq::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
  ASSERT_EQ(MC2.validation(), true);
  MC2.pre_processing();
  MC2.run();
  MC2.post_processing();
  EXPECT_LE(abs(etalon - result_seq) / etalon, 1e-1);
}

TEST(grudzin_k_monte_carlo_seq, Test_3Dim) {
  const int dimensions = 3;
  int N = 10000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * (x[0] + x[1] + x[2]) + 1.0; }
  };
  std::vector<double> dim = grudzin_k_montecarlo_seq::GenDimDistr(dimensions);
  double etalon = grudzin_k_montecarlo_seq::CalcEtalon(dim);
  double result_seq = 0;
  std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  MC2_Data->inputs_count.emplace_back(dimensions);
  MC2_Data->inputs_count.emplace_back(1);
  MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
  MC2_Data->outputs_count.emplace_back(1);

  grudzin_k_montecarlo_seq::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
  ASSERT_EQ(MC2.validation(), true);
  MC2.pre_processing();
  MC2.run();
  MC2.post_processing();
  EXPECT_LE(abs(etalon - result_seq) / etalon, 1e-1);
}

TEST(grudzin_k_monte_carlo_seq, Test_3Dim_UnusualFunction) {
  const int dimensions = 3;
  int N = 10000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    return std::exp(x[0] + x[1] + x[2]);
  };
  // precalculated result by online calculator
  std::vector<double> dim = {0, 1, 0, 1, 0, 1};
  double etalon = 5.07321411177285;
  double result_seq = 0;
  std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  MC2_Data->inputs_count.emplace_back(dimensions);
  MC2_Data->inputs_count.emplace_back(1);
  MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
  MC2_Data->outputs_count.emplace_back(1);

  grudzin_k_montecarlo_seq::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
  ASSERT_EQ(MC2.validation(), true);
  MC2.pre_processing();
  MC2.run();
  MC2.post_processing();
  EXPECT_LE(abs(etalon - result_seq) / etalon, 1e-1);
}

TEST(grudzin_k_monte_carlo_seq, Test_3Dim_2k) {
  const int dimensions = 3;
  int N = 1 << 13;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * (x[0] + x[1] + x[2]) + 1.0; }
  };
  std::vector<double> dim = grudzin_k_montecarlo_seq::GenDimDistr(dimensions);
  double etalon = grudzin_k_montecarlo_seq::CalcEtalon(dim);
  double result_seq = 0;

  std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  MC2_Data->inputs_count.emplace_back(dimensions);
  MC2_Data->inputs_count.emplace_back(1);
  MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
  MC2_Data->outputs_count.emplace_back(1);

  grudzin_k_montecarlo_seq::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
  ASSERT_EQ(MC2.validation(), true);
  MC2.pre_processing();
  MC2.run();
  MC2.post_processing();
  EXPECT_LE(abs(etalon - result_seq) / etalon, 1e-1);
}

TEST(grudzin_k_monte_carlo_seq, Test_3Dim_prime) {
  const int dimensions = 3;
  int N = 100003;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * (x[0] + x[1] + x[2]) + 1.0; }
  };
  std::vector<double> dim = grudzin_k_montecarlo_seq::GenDimDistr(dimensions);
  double etalon = grudzin_k_montecarlo_seq::CalcEtalon(dim);
  double result_seq = 0;

  std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  MC2_Data->inputs_count.emplace_back(dimensions);
  MC2_Data->inputs_count.emplace_back(1);
  MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
  MC2_Data->outputs_count.emplace_back(1);

  grudzin_k_montecarlo_seq::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
  ASSERT_EQ(MC2.validation(), true);
  MC2.pre_processing();
  MC2.run();
  MC2.post_processing();
  EXPECT_LE(abs(etalon - result_seq) / etalon, 1e-1);
}

TEST(grudzin_k_monte_carlo_seq, Empty_Out) {
  const int dimensions = 3;
  int N = 10000;
  auto f = [](std::array<double, 3> &x) -> double { return std::pow(x[0], 3) + std::pow(x[1] + x[2], 2) + 2.0 * x[2]; };
  std::vector<double> dim = grudzin_k_montecarlo_seq::GenDimDistr(dimensions);

  std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();

  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  MC2_Data->inputs_count.emplace_back(dimensions);
  MC2_Data->inputs_count.emplace_back(1);
  grudzin_k_montecarlo_seq::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
  ASSERT_EQ(MC2.validation(), false);
}

TEST(grudzin_k_monte_carlo_seq, Forget_One_Val) {
  const int dimensions = 3;
  auto f = [](std::array<double, 3> &x) -> double { return std::pow(x[0], 3) + std::pow(x[1] + x[2], 2) + 2.0 * x[2]; };
  std::vector<double> dim = grudzin_k_montecarlo_seq::GenDimDistr(dimensions);
  double result_seq = 0;

  std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();

  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
  MC2_Data->inputs_count.emplace_back(dimensions);
  MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
  MC2_Data->outputs_count.emplace_back(1);
  grudzin_k_montecarlo_seq::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
  ASSERT_EQ(MC2.validation(), false);
}

TEST(grudzin_k_monte_carlo_seq, IC_NEQ_DIM) {
  const int dimensions = 3;
  int N = 10000;
  auto f = [](std::array<double, 3> &x) -> double { return std::pow(x[0], 3) + std::pow(x[1] + x[2], 2) + 2.0 * x[2]; };
  std::vector<double> dim = grudzin_k_montecarlo_seq::GenDimDistr(dimensions - 1);
  double result_seq = 0;

  std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();

  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
  MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  MC2_Data->inputs_count.emplace_back(dimensions - 1);
  MC2_Data->inputs_count.emplace_back(1);
  MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
  MC2_Data->outputs_count.emplace_back(1);

  grudzin_k_montecarlo_seq::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
  ASSERT_EQ(MC2.validation(), false);
}