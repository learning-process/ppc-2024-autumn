#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/grudzin_k_monte_carlo/include/gmc_include.hpp"

namespace grudzin_k_montecarlo_mpi {
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
}  // namespace grudzin_k_montecarlo_mpi

TEST(grudzin_k_monte_carlo_mpi, Test_1Dim) {
  boost::mpi::communicator world;
  const int dimensions = 1;
  int N = 10000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * x[0] + 1.0; }
  };
  std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> dim = grudzin_k_montecarlo_mpi::GenDimDistr(dimensions, -1.0, 1.0);
  double result_par = 0;
  double result_seq = 0;

  if (world.rank() == 0) {
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->inputs_count.emplace_back(1);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);
  }
  grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions> MC1(MC1_Data, f);
  ASSERT_EQ(MC1.validation(), true);
  MC1.pre_processing();
  MC1.run();
  MC1.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC2_Data->inputs_count.emplace_back(dimensions);
    MC2_Data->inputs_count.emplace_back(1);
    MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
    MC2_Data->outputs_count.emplace_back(1);

    grudzin_k_montecarlo_mpi::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
    ASSERT_EQ(MC2.validation(), true);
    MC2.pre_processing();
    MC2.run();
    MC2.post_processing();
    EXPECT_LE(abs(result_par - result_seq) / result_seq, 1e-1);
  }
}

TEST(grudzin_k_monte_carlo_mpi, Test_2Dim) {
  boost::mpi::communicator world;
  const int dimensions = 2;
  int N = 10000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * (x[0] + x[1]) + 1.0; }
  };
  std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> dim = grudzin_k_montecarlo_mpi::GenDimDistr(dimensions, -2.0, 2.0);
  double result_par = 0;
  double result_seq = 0;

  if (world.rank() == 0) {
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->inputs_count.emplace_back(1);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);
  }
  grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions> MC1(MC1_Data, f);
  ASSERT_EQ(MC1.validation(), true);
  MC1.pre_processing();
  MC1.run();
  MC1.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC2_Data->inputs_count.emplace_back(dimensions);
    MC2_Data->inputs_count.emplace_back(1);
    MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
    MC2_Data->outputs_count.emplace_back(1);

    grudzin_k_montecarlo_mpi::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
    ASSERT_EQ(MC2.validation(), true);
    MC2.pre_processing();
    MC2.run();
    MC2.post_processing();
    EXPECT_LE(abs(result_par - result_seq) / result_seq, 1e-1);
  }
}

TEST(grudzin_k_monte_carlo_mpi, Test_3Dim) {
  boost::mpi::communicator world;
  const int dimensions = 3;
  int N = 10000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * (x[0] + x[1] + x[2]) + 1.0; }
  };
  std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> dim = grudzin_k_montecarlo_mpi::GenDimDistr(dimensions, -3.0, 3.0);
  double result_par = 0;
  double result_seq = 0;

  if (world.rank() == 0) {
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->inputs_count.emplace_back(1);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);
  }
  grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions> MC1(MC1_Data, f);
  ASSERT_EQ(MC1.validation(), true);
  MC1.pre_processing();
  MC1.run();
  MC1.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC2_Data->inputs_count.emplace_back(dimensions);
    MC2_Data->inputs_count.emplace_back(1);
    MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
    MC2_Data->outputs_count.emplace_back(1);

    grudzin_k_montecarlo_mpi::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
    ASSERT_EQ(MC2.validation(), true);
    MC2.pre_processing();
    MC2.run();
    MC2.post_processing();
    EXPECT_LE(abs(result_par - result_seq) / result_seq, 1e-1);
  }
}

TEST(grudzin_k_monte_carlo_mpi, Test_3Dim_2k) {
  boost::mpi::communicator world;
  const int dimensions = 3;
  int N = 1 << 13;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * (x[0] + x[1] + x[2]) + 1.0; }
  };
  std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> dim = grudzin_k_montecarlo_mpi::GenDimDistr(dimensions, -4.0, 4.0);
  double result_par = 0;
  double result_seq = 0;

  if (world.rank() == 0) {
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->inputs_count.emplace_back(1);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);
  }
  grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions> MC1(MC1_Data, f);
  ASSERT_EQ(MC1.validation(), true);
  MC1.pre_processing();
  MC1.run();
  MC1.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC2_Data->inputs_count.emplace_back(dimensions);
    MC2_Data->inputs_count.emplace_back(1);
    MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
    MC2_Data->outputs_count.emplace_back(1);

    grudzin_k_montecarlo_mpi::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
    ASSERT_EQ(MC2.validation(), true);
    MC2.pre_processing();
    MC2.run();
    MC2.post_processing();
    EXPECT_LE(abs(result_par - result_seq) / result_seq, 1e-1);
  }
}

TEST(grudzin_k_monte_carlo_mpi, Test_3Dim_prime) {
  boost::mpi::communicator world;
  const int dimensions = 3;
  int N = 100003;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * (x[0] + x[1] + x[2]) + 1.0; }
  };
  std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> dim = grudzin_k_montecarlo_mpi::GenDimDistr(dimensions, -5.0, 5.0);
  double result_par = 0;
  double result_seq = 0;

  if (world.rank() == 0) {
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->inputs_count.emplace_back(1);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);
  }
  grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions> MC1(MC1_Data, f);
  ASSERT_EQ(MC1.validation(), true);
  MC1.pre_processing();
  MC1.run();
  MC1.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC2_Data->inputs_count.emplace_back(dimensions);
    MC2_Data->inputs_count.emplace_back(1);
    MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
    MC2_Data->outputs_count.emplace_back(1);

    grudzin_k_montecarlo_mpi::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
    ASSERT_EQ(MC2.validation(), true);
    MC2.pre_processing();
    MC2.run();
    MC2.post_processing();
    EXPECT_LE(abs(result_par - result_seq) / result_seq, 1e-1);
  }
}

TEST(grudzin_k_monte_carlo_mpi, Test_EXP) {
  boost::mpi::communicator world;
  const int dimensions = 3;
  int N = 10000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    return std::exp(x[0] + x[1] + x[2]);
  };
  std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> dim = grudzin_k_montecarlo_mpi::GenDimDistr(dimensions, -6.0, 6.0);
  double result_par = 0;
  double result_seq = 0;

  if (world.rank() == 0) {
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->inputs_count.emplace_back(1);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);
  }
  grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions> MC1(MC1_Data, f);
  ASSERT_EQ(MC1.validation(), true);
  MC1.pre_processing();
  MC1.run();
  MC1.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC2_Data->inputs_count.emplace_back(dimensions);
    MC2_Data->inputs_count.emplace_back(1);
    MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
    MC2_Data->outputs_count.emplace_back(1);

    grudzin_k_montecarlo_mpi::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
    ASSERT_EQ(MC2.validation(), true);
    MC2.pre_processing();
    MC2.run();
    MC2.post_processing();
    EXPECT_LE(abs(result_par - result_seq) / result_seq, 1e-1);
  }
}

TEST(grudzin_k_monte_carlo_mpi, Test_Poly) {
  boost::mpi::communicator world;
  const int dimensions = 3;

  int N = 100000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    return std::pow(x[0], 3) + std::pow(x[1] + x[2], 2) + 2.0 * x[2];
  };
  std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> dim = grudzin_k_montecarlo_mpi::GenDimDistr(dimensions, -7.0, 7.0);
  double result_par = 0;
  double result_seq = 0;

  if (world.rank() == 0) {
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->inputs_count.emplace_back(1);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);
  }
  grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions> MC1(MC1_Data, f);
  ASSERT_EQ(MC1.validation(), true);
  MC1.pre_processing();
  MC1.run();
  MC1.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC2_Data->inputs_count.emplace_back(dimensions);
    MC2_Data->inputs_count.emplace_back(1);
    MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
    MC2_Data->outputs_count.emplace_back(1);

    grudzin_k_montecarlo_mpi::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
    ASSERT_EQ(MC2.validation(), true);
    MC2.pre_processing();
    MC2.run();
    MC2.post_processing();
    EXPECT_LE(abs(result_par - result_seq) / result_seq, 1e-1);
  }
}

TEST(grudzin_k_monte_carlo_mpi, Empty_Out) {
  boost::mpi::communicator world;
  const int dimensions = 3;
  int N = 10000;
  auto f = [](std::array<double, 3> &x) -> double { return std::pow(x[0], 3) + std::pow(x[1] + x[2], 2) + 2.0 * x[2]; };
  std::vector<double> dim = grudzin_k_montecarlo_mpi::GenDimDistr(dimensions);

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
    std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->inputs_count.emplace_back(1);

    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC2_Data->inputs_count.emplace_back(dimensions);
    MC2_Data->inputs_count.emplace_back(1);
    grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions> MC1(MC1_Data, f);
    grudzin_k_montecarlo_mpi::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
    ASSERT_EQ(MC1.validation(), false);
    ASSERT_EQ(MC2.validation(), false);
  }
}

TEST(grudzin_k_monte_carlo_mpi, Forget_One_Val) {
  boost::mpi::communicator world;
  const int dimensions = 3;
  auto f = [](std::array<double, 3> &x) -> double { return std::pow(x[0], 3) + std::pow(x[1] + x[2], 2) + 2.0 * x[2]; };
  std::vector<double> dim = grudzin_k_montecarlo_mpi::GenDimDistr(dimensions);
  double result_par = 0;
  double result_seq = 0;

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
    std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);

    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC2_Data->inputs_count.emplace_back(dimensions);
    MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
    MC2_Data->outputs_count.emplace_back(1);
    grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions> MC1(MC1_Data, f);
    grudzin_k_montecarlo_mpi::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
    ASSERT_EQ(MC1.validation(), false);
    ASSERT_EQ(MC2.validation(), false);
  }
}

TEST(grudzin_k_monte_carlo_mpi, IC_NEQ_DIM) {
  boost::mpi::communicator world;
  const int dimensions = 3;
  int N = 10000;
  auto f = [](std::array<double, 3> &x) -> double { return std::pow(x[0], 3) + std::pow(x[1] + x[2], 2) + 2.0 * x[2]; };
  std::vector<double> dim = grudzin_k_montecarlo_mpi::GenDimDistr(dimensions - 1);
  double result_par = 0;
  double result_seq = 0;

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
    std::shared_ptr<ppc::core::TaskData> MC2_Data = std::make_shared<ppc::core::TaskData>();
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions - 1);
    MC1_Data->inputs_count.emplace_back(1);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);

    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC2_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC2_Data->inputs_count.emplace_back(dimensions - 1);
    MC2_Data->inputs_count.emplace_back(1);
    MC2_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_seq));
    MC2_Data->outputs_count.emplace_back(1);

    grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions> MC1(MC1_Data, f);
    grudzin_k_montecarlo_mpi::MonteCarloSeq<dimensions> MC2(MC2_Data, f);
    ASSERT_EQ(MC1.validation(), false);
    ASSERT_EQ(MC2.validation(), false);
  }
}