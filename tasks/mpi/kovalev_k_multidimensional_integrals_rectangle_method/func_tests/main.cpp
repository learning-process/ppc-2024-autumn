#include <gtest/gtest.h>

#include <numbers>

#include "mpi/kovalev_k_multidimensional_integrals_rectangle_method/include/header.hpp"

namespace kovalev_k_multidimensional_integrals_rectangle_method_mpi {
double area(std::vector<double> &arguments) { return 1.0; }

double f1(std::vector<double> &arguments) { return arguments.at(0); }
double f1cos(std::vector<double> &arguments) { return std::cos(arguments.at(0)); }
double f1Euler(std::vector<double> &arguments) { return 2 * std::cos(arguments.at(0)) * std::sin(arguments.at(0)); }
double f2(std::vector<double> &arguments) { return arguments.at(0) * arguments.at(1); }
double f2advanced(std::vector<double> &arguments) { return std::tan(arguments.at(0)) * std::atan(arguments.at(1)); }
double f3(std::vector<double> &arguments) { return arguments.at(0) * arguments.at(1) * arguments.at(2); }
double f3advanced(std::vector<double> &arguments) {
  return std::sin(arguments.at(0)) * std::tan(arguments.at(1)) * std::log(arguments.at(2));
}
}  // namespace kovalev_k_multidimensional_integrals_rectangle_method_mpi

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, invalid_integration_step) {
  std::vector<std::pair<double, double>> lims;
  double h = 111.1;
  std::vector<double> out;
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(lims.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      tmpPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f1);
  if (world.rank() == 0) {
    ASSERT_FALSE(tmpTaskPar.validation());
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, zero_length) {
  std::vector<std::pair<double, double>> lims;
  double h = 0.001;
  std::vector<double> out;
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(lims.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      tmpPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f1);
  if (world.rank() == 0) {
    ASSERT_FALSE(tmpTaskPar.validation());
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, incorrect_output) {
  std::vector<std::pair<double, double>> lims(1);
  double h = 0.001;
  std::vector<double> out(2);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f1);
  if (world.rank() == 0) {
    ASSERT_FALSE(tmpTaskPar.validation());
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, minus_05pi_05pi_cos) {
  const size_t dim = 1;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = -0.5 * std::numbers::pi;
  lims[0].second = 0.5 * std::numbers::pi;
  double h = 0.0005;
  double eps = 1e-4;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(lims.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      tmpPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f1cos);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(2.0, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, Eulers_integral) {
  const size_t dim = 1;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = 0;
  lims[0].second = 0.5 * std::numbers::pi;
  double h = 0.0005;
  double eps = 1e-3;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f1Euler);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(1.0, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, 1x1_area) {
  const size_t dim = 2;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = lims[1].first = 0.0;
  lims[0].second = lims[1].second = 1.0;
  double h = 0.0005;
  double eps = 1e-3;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::area);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(1.0, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, 1x1_xy) {
  const size_t dim = 2;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = lims[1].first = 0.0;
  lims[0].second = lims[1].second = 1.0;
  double h = 0.0005;
  double eps = 5e-4;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f2);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(0.25, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, minus1_0xminus1_0_tg_x_arctan_y) {
  const size_t dim = 2;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = lims[1].first = 0.0;
  lims[0].second = lims[1].second = 1.0;
  double h = 0.0005;
  double eps = 5e-4;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f2advanced);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(0.270152023066961, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, 2x2_area) {
  const size_t dim = 2;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = lims[1].first = 1.0;
  lims[0].second = lims[1].second = 2.0;
  double h = 0.0005;
  double eps = 1e-3;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::area);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(1.0, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, 2_5x1_4_area) {
  const size_t dim = 2;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = 2.0;
  lims[0].second = 5.0;
  lims[1].first = 1.0;
  lims[1].second = 4.0;
  double h = 0.005;
  double eps = 1e-4;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::area);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(9.0, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, minus1_0xminus1_0_area) {
  const size_t dim = 2;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = -1.0;
  lims[0].second = 0.0;
  lims[1].first = -1.0;
  lims[1].second = 0.0;
  double h = 0.0005;
  double eps = 1e-3;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::area);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(1.0, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, minus1_0xminus1_0_xy) {
  const size_t dim = 2;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = -1.0;
  lims[0].second = 0.0;
  lims[1].first = -1.0;
  lims[1].second = 0.0;
  double h = 0.0005;
  double eps = 1e-4;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f2);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(0.25, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, minus03_0_x_15_17_x_2_21_area) {
  const size_t dim = 3;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = -0.3;
  lims[0].second = 0.0;
  lims[1].first = 1.5;
  lims[1].second = 1.7;
  lims[2].first = 2.0;
  lims[2].second = 2.1;
  double h = 0.001;
  double eps = 1e-4;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::area);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(0.006, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, 09_1_x_15_17_x_18_2_xyz) {
  const size_t dim = 3;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = 0.9;
  lims[0].second = 1.0;
  lims[1].first = 1.5;
  lims[1].second = 1.7;
  lims[2].first = 1.8;
  lims[2].second = 2.0;
  double h = 0.001;
  double eps = 1e-3;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f3);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(0.011552, out[0], eps);
  }
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_mpi, 08_1_x_19_2_x_29_3_sinx_tgy_lnz) {
  const size_t dim = 3;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = 0.8;
  lims[0].second = 1.0;
  lims[1].first = 1.9;
  lims[1].second = 2.0;
  lims[2].first = 2.9;
  lims[2].second = 3.0;
  double h = 0.005;
  double eps = 1e-3;
  std::vector<double> out(1);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> TaskPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskPar->inputs_count.emplace_back(lims.size());
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
    TaskPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
    TaskPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    TaskPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar tmpTaskPar(
      TaskPar, kovalev_k_multidimensional_integrals_rectangle_method_mpi::f3advanced);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    ASSERT_NEAR(-0.00427191467841401, out[0], eps);
  }
}