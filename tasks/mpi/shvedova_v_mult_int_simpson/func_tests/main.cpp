#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/shvedova_v_mult_int_simpson/include/ops_mpi.hpp"

namespace shvedova_v_mult_int_simpson_mpi {

void runTestParallel(std::vector<std::pair<double, double>> limits,
                     std::function<double(std::vector<double>& args)> func, double precision = 1e-4) {
  boost::mpi::communicator world;
  double out = 0;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&precision));
    taskDataPar->inputs_count.emplace_back(limits.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  shvedova_v_mult_int_simpson_mpi::SimpsonMultIntParallel testTaskParallel(taskDataPar, func);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    double seq_out = 0;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&precision));
    taskDataSeq->inputs_count.emplace_back(limits.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&seq_out));
    taskDataSeq->outputs_count.emplace_back(1);

    shvedova_v_mult_int_simpson_mpi::SimpsonMultIntSequential testTaskSequential(taskDataSeq, func);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_NEAR(seq_out, out, precision);
  }
}

double linearFunction(std::vector<double>& args) { return args[0]; }
double sumTwoVariables(std::vector<double>& args) { return args[0] + args[1]; }
double quadraticFunction(std::vector<double>& args) { return args[0] * args[0]; }
double productTwoVariables(std::vector<double>& args) { return args[0] * args[1]; }
double trigonometricFunction(std::vector<double>& args) { return std::sin(args[0]) + std::cos(args[1]); }
double exponentialFunction(std::vector<double>& args) { return exp(args[0]) - args[1]; }
double tripleLinearFunction(std::vector<double>& args) { return args[0] + args[1] + args[2]; }
double tripleProductFunction(std::vector<double>& args) { return args[0] * args[1] * args[2]; }

}  // namespace shvedova_v_mult_int_simpson_mpi

TEST(shvedova_v_mult_int_simpson_mpi, LinearFunction_SingleVariable) {
  shvedova_v_mult_int_simpson_mpi::runTestParallel({{0, 2}}, shvedova_v_mult_int_simpson_mpi::linearFunction);
}

TEST(shvedova_v_mult_int_simpson_mpi, LinearFunction_MultipleVariables) {
  shvedova_v_mult_int_simpson_mpi::runTestParallel({{0, 2}, {0, 2}}, shvedova_v_mult_int_simpson_mpi::linearFunction);
}

TEST(shvedova_v_mult_int_simpson_mpi, SumTwoVariables_TwoDimensions) {
  shvedova_v_mult_int_simpson_mpi::runTestParallel({{0, 2}, {0, 2}}, shvedova_v_mult_int_simpson_mpi::sumTwoVariables);
}

TEST(shvedova_v_mult_int_simpson_mpi, QuadraticFunction_OneDimension) {
  shvedova_v_mult_int_simpson_mpi::runTestParallel({{0, 3}}, shvedova_v_mult_int_simpson_mpi::quadraticFunction);
}

TEST(shvedova_v_mult_int_simpson_mpi, ProductTwoVariables_TwoDimensions) {
  shvedova_v_mult_int_simpson_mpi::runTestParallel({{1, 3}, {2, 4}},
                                                   shvedova_v_mult_int_simpson_mpi::productTwoVariables);
}

TEST(shvedova_v_mult_int_simpson_mpi, TrigonometricFunction_TwoDimensions) {
  shvedova_v_mult_int_simpson_mpi::runTestParallel({{0, M_PI}, {0, M_PI / 2}},
                                                   shvedova_v_mult_int_simpson_mpi::trigonometricFunction);
}

TEST(shvedova_v_mult_int_simpson_mpi, ExponentialFunction_TwoDimensions) {
  shvedova_v_mult_int_simpson_mpi::runTestParallel({{0, 1}, {0, 1}},
                                                   shvedova_v_mult_int_simpson_mpi::exponentialFunction);
}

TEST(shvedova_v_mult_int_simpson_mpi, TripleLinearFunction) {
  shvedova_v_mult_int_simpson_mpi::runTestParallel({{0, 1}, {0, 1}, {0, 1}},
                                                   shvedova_v_mult_int_simpson_mpi::tripleLinearFunction);
}

TEST(shvedova_v_mult_int_simpson_mpi, TripleProductFunction) {
  shvedova_v_mult_int_simpson_mpi::runTestParallel({{1, 2}, {1, 2}, {1, 2}},
                                                   shvedova_v_mult_int_simpson_mpi::tripleProductFunction);
}