// mpi func tests rectangle method
#include <gtest/gtest.h>

#include "mpi/rezantseva_a_rectangle_method/include/ops_mpi_rez_a.hpp"

TEST(rezantseva_a_rectangle_method_mpi, check_1_dimension_integral) {
  boost::mpi::communicator world;
  double error = 0.0001;
  std::vector<double> out(1, 0.0);
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return x[0] * x[0];  // x^2
  };

  int n = 1;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {4, 10};
  distrib[0] = 1000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataMPI->inputs_count.emplace_back(bounds.size());

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataMPI->inputs_count.emplace_back(distrib.size());

    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_rectangle_method_mpi::RectangleMethodMPI RectangleMethodMPI(taskDataMPI, function);
  ASSERT_EQ(RectangleMethodMPI.validation(), true);
  RectangleMethodMPI.pre_processing();
  RectangleMethodMPI.run();
  RectangleMethodMPI.post_processing();

  std::vector<double> seq_out(1, 0.0);
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataSeq->inputs_count.emplace_back(bounds.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataSeq->inputs_count.emplace_back(distrib.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    rezantseva_a_rectangle_method_mpi::RectangleMethodSequential RectangleMethodSeq(taskDataSeq, function);
    ASSERT_EQ(RectangleMethodSeq.validation(), true);
    RectangleMethodSeq.pre_processing();
    RectangleMethodSeq.run();
    RectangleMethodSeq.post_processing();

    ASSERT_NEAR(seq_out[0], out[0], error);
  }
}

TEST(rezantseva_a_rectangle_method_mpi, check_linear_func_2_dimension) {
  boost::mpi::communicator world;
  double error = 0.0001;
  std::vector<double> out(1, 0.0);

  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return x[0] * x[0] - 2 * x[1];  // x^2-2y
  };

  int n = 2;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    bounds[0] = {4, 10};
    bounds[1] = {1, 56};
    distrib[0] = 1000;
    distrib[1] = 1000;
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataMPI->inputs_count.emplace_back(bounds.size());

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataMPI->inputs_count.emplace_back(distrib.size());

    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_rectangle_method_mpi::RectangleMethodMPI RectangleMethodMPI(taskDataMPI, function);
  ASSERT_EQ(RectangleMethodMPI.validation(), true);
  RectangleMethodMPI.pre_processing();
  RectangleMethodMPI.run();
  RectangleMethodMPI.post_processing();

  std::vector<double> seq_out(1, 0.0);
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataSeq->inputs_count.emplace_back(bounds.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataSeq->inputs_count.emplace_back(distrib.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    rezantseva_a_rectangle_method_mpi::RectangleMethodSequential RectangleMethodSeq(taskDataSeq, function);
    ASSERT_EQ(RectangleMethodSeq.validation(), true);
    RectangleMethodSeq.pre_processing();
    RectangleMethodSeq.run();
    RectangleMethodSeq.post_processing();

    ASSERT_NEAR(seq_out[0], out[0], error);
  }
}

TEST(rezantseva_a_rectangle_method_mpi, check_1_dimension_integral_sin) {
  boost::mpi::communicator world;
  double error = 0.0001;
  std::vector<double> out(1, 0.0);
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return std::sin(x[0]) + x[0] * x[0] * x[0];  // sinx+x^3
  };

  int n = 1;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {4, 10};
  distrib[0] = 1000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataMPI->inputs_count.emplace_back(bounds.size());

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataMPI->inputs_count.emplace_back(distrib.size());

    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_rectangle_method_mpi::RectangleMethodMPI RectangleMethodMPI(taskDataMPI, function);
  ASSERT_EQ(RectangleMethodMPI.validation(), true);
  RectangleMethodMPI.pre_processing();
  RectangleMethodMPI.run();
  RectangleMethodMPI.post_processing();

  std::vector<double> seq_out(1, 0.0);
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataSeq->inputs_count.emplace_back(bounds.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataSeq->inputs_count.emplace_back(distrib.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    rezantseva_a_rectangle_method_mpi::RectangleMethodSequential RectangleMethodSeq(taskDataSeq, function);
    ASSERT_EQ(RectangleMethodSeq.validation(), true);
    RectangleMethodSeq.pre_processing();
    RectangleMethodSeq.run();
    RectangleMethodSeq.post_processing();

    ASSERT_NEAR(seq_out[0], out[0], error);
  }
}

TEST(rezantseva_a_rectangle_method_mpi, check_2_dimension_integral_sin) {
  boost::mpi::communicator world;
  double error = 0.0001;
  std::vector<double> out(1, 0.0);
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return std::sin(x[0]) + x[0] * x[0] * x[1];  // sinx + y*x^2
  };

  int n = 2;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {-3, 10};
  bounds[1] = {-7, 25};
  distrib[0] = 100;
  distrib[1] = 100;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataMPI->inputs_count.emplace_back(bounds.size());

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataMPI->inputs_count.emplace_back(distrib.size());

    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_rectangle_method_mpi::RectangleMethodMPI RectangleMethodMPI(taskDataMPI, function);
  ASSERT_EQ(RectangleMethodMPI.validation(), true);
  RectangleMethodMPI.pre_processing();
  RectangleMethodMPI.run();
  RectangleMethodMPI.post_processing();

  std::vector<double> seq_out(1, 0.0);
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataSeq->inputs_count.emplace_back(bounds.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataSeq->inputs_count.emplace_back(distrib.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    rezantseva_a_rectangle_method_mpi::RectangleMethodSequential RectangleMethodSeq(taskDataSeq, function);
    ASSERT_EQ(RectangleMethodSeq.validation(), true);
    RectangleMethodSeq.pre_processing();
    RectangleMethodSeq.run();
    RectangleMethodSeq.post_processing();

    ASSERT_NEAR(seq_out[0], out[0], error);
  }
}

TEST(rezantseva_a_rectangle_method_mpi, check_3_dimension_integral) {
  boost::mpi::communicator world;
  double error = 0.0001;
  std::vector<double> out(1, 0.0);
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return x[0] + x[1] + x[2];  // xyz
  };

  int n = 3;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {4, 100};
  bounds[1] = {1, 156};
  bounds[2] = {6, 249};
  distrib[0] = 100;
  distrib[1] = 100;
  distrib[2] = 100;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataMPI->inputs_count.emplace_back(bounds.size());

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataMPI->inputs_count.emplace_back(distrib.size());

    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_rectangle_method_mpi::RectangleMethodMPI RectangleMethodMPI(taskDataMPI, function);
  ASSERT_EQ(RectangleMethodMPI.validation(), true);
  RectangleMethodMPI.pre_processing();
  RectangleMethodMPI.run();
  RectangleMethodMPI.post_processing();

  std::vector<double> seq_out(1, 0.0);
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataSeq->inputs_count.emplace_back(bounds.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataSeq->inputs_count.emplace_back(distrib.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    rezantseva_a_rectangle_method_mpi::RectangleMethodSequential RectangleMethodSeq(taskDataSeq, function);
    ASSERT_EQ(RectangleMethodSeq.validation(), true);
    RectangleMethodSeq.pre_processing();
    RectangleMethodSeq.run();
    RectangleMethodSeq.post_processing();

    ASSERT_NEAR(seq_out[0], out[0], error);
  }
}

TEST(rezantseva_a_rectangle_method_mpi, check_3_dimension_integral_with_exp) {
  boost::mpi::communicator world;
  double error = 0.0001;
  std::vector<double> out(1, 0.0);
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return std::exp(x[0] + 2 * x[1]) - 2 * std::cos(x[2]);  // exp(x+2y) - 2cosz + sqrt(4d)
  };

  int n = 3;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {4, 10};
  bounds[1] = {-7, 3};
  bounds[2] = {3, 8};

  distrib[0] = 100;
  distrib[1] = 100;
  distrib[2] = 100;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataMPI->inputs_count.emplace_back(bounds.size());

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataMPI->inputs_count.emplace_back(distrib.size());

    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_rectangle_method_mpi::RectangleMethodMPI RectangleMethodMPI(taskDataMPI, function);
  ASSERT_EQ(RectangleMethodMPI.validation(), true);
  RectangleMethodMPI.pre_processing();
  RectangleMethodMPI.run();
  RectangleMethodMPI.post_processing();

  std::vector<double> seq_out(1, 0.0);
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataSeq->inputs_count.emplace_back(bounds.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataSeq->inputs_count.emplace_back(distrib.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    rezantseva_a_rectangle_method_mpi::RectangleMethodSequential RectangleMethodSeq(taskDataSeq, function);
    ASSERT_EQ(RectangleMethodSeq.validation(), true);
    RectangleMethodSeq.pre_processing();
    RectangleMethodSeq.run();
    RectangleMethodSeq.post_processing();

    ASSERT_NEAR(seq_out[0], out[0], error);
  }
}

TEST(rezantseva_a_rectangle_method_mpi, check_3_dimension_integral_with_log) {
  boost::mpi::communicator world;
  double error = 0.0001;
  std::vector<double> out(1, 0.0);
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return (std::log10(2 * x[0] * x[0]) + sqrt(x[2]) + 5 * x[1]);
  };

  int n = 3;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {0, 1};
  bounds[1] = {-13, 5};
  bounds[2] = {3, 7};
  distrib[0] = 100;
  distrib[1] = 100;
  distrib[2] = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataMPI->inputs_count.emplace_back(bounds.size());

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataMPI->inputs_count.emplace_back(distrib.size());

    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_rectangle_method_mpi::RectangleMethodMPI RectangleMethodMPI(taskDataMPI, function);
  ASSERT_EQ(RectangleMethodMPI.validation(), true);
  RectangleMethodMPI.pre_processing();
  RectangleMethodMPI.run();
  RectangleMethodMPI.post_processing();

  std::vector<double> seq_out(1, 0.0);
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataSeq->inputs_count.emplace_back(bounds.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataSeq->inputs_count.emplace_back(distrib.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    rezantseva_a_rectangle_method_mpi::RectangleMethodSequential RectangleMethodSeq(taskDataSeq, function);
    ASSERT_EQ(RectangleMethodSeq.validation(), true);
    RectangleMethodSeq.pre_processing();
    RectangleMethodSeq.run();
    RectangleMethodSeq.post_processing();

    ASSERT_NEAR(seq_out[0], out[0], error);
  }
}

TEST(rezantseva_a_rectangle_method_mpi, check_3_dimension_integral_with_cos) {
  boost::mpi::communicator world;
  double error = 0.0001;
  std::vector<double> out(1, 0.0);
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return (std::log10(2 * x[0] * x[2]) + std::sin(x[1]) + 5 * std::cos(x[2]));
  };

  int n = 3;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {0, 1};
  bounds[1] = {-13, 5};
  bounds[2] = {3, 7};
  distrib[0] = 50;
  distrib[1] = 50;
  distrib[2] = 50;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataMPI->inputs_count.emplace_back(bounds.size());

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataMPI->inputs_count.emplace_back(distrib.size());

    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }
  // Create Task
  rezantseva_a_rectangle_method_mpi::RectangleMethodMPI RectangleMethodMPI(taskDataMPI, function);
  ASSERT_EQ(RectangleMethodMPI.validation(), true);
  RectangleMethodMPI.pre_processing();
  RectangleMethodMPI.run();
  RectangleMethodMPI.post_processing();

  std::vector<double> seq_out(1, 0.0);
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataSeq->inputs_count.emplace_back(bounds.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataSeq->inputs_count.emplace_back(distrib.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    rezantseva_a_rectangle_method_mpi::RectangleMethodSequential RectangleMethodSeq(taskDataSeq, function);
    ASSERT_EQ(RectangleMethodSeq.validation(), true);
    RectangleMethodSeq.pre_processing();
    RectangleMethodSeq.run();
    RectangleMethodSeq.post_processing();

    ASSERT_NEAR(seq_out[0], out[0], error);
  }
}