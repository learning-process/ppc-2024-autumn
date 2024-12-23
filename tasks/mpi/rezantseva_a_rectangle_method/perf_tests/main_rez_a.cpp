// mpi perf tests rectangle method
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/rezantseva_a_rectangle_method/include/ops_mpi_rez_a.hpp"

TEST(rezantseva_a_rectangle_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  double error = 0.0001;
  std::vector<double> out(1, 0.0);
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return x[0] * x[0] - 2 * x[1] + 8 * x[2];
  };

  int n = 3;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {-6, 10};
  bounds[1] = {5, 27};
  bounds[2] = {15, 32};
  distrib[0] = 500;
  distrib[1] = 250;
  distrib[2] = 150;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataMPI->inputs_count.emplace_back(bounds.size());

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataMPI->inputs_count.emplace_back(distrib.size());

    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  auto rectanglMethodParallel =
      std::make_shared<rezantseva_a_rectangle_method_mpi::RectangleMethodMPI>(taskDataMPI, function);
  ASSERT_EQ(rectanglMethodParallel->validation(), true);
  rectanglMethodParallel->pre_processing();
  rectanglMethodParallel->run();
  rectanglMethodParallel->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  //  Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(rectanglMethodParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

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

TEST(rezantseva_a_rectangle_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  double error = 0.0001;
  std::vector<double> out(1, 0.0);
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return x[0] * x[0] - 2 * x[1] + 8 * x[2];
  };

  int n = 3;
  std::vector<std::pair<double, double>> bounds(n);
  std::vector<int> distrib(n);

  bounds[0] = {-6, 10};
  bounds[1] = {5, 27};
  bounds[2] = {15, 32};
  distrib[0] = 500;
  distrib[1] = 250;
  distrib[2] = 150;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&bounds));
    taskDataMPI->inputs_count.emplace_back(bounds.size());

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(&distrib));
    taskDataMPI->inputs_count.emplace_back(distrib.size());

    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  auto rectanglMethodParallel =
      std::make_shared<rezantseva_a_rectangle_method_mpi::RectangleMethodMPI>(taskDataMPI, function);
  ASSERT_EQ(rectanglMethodParallel->validation(), true);
  rectanglMethodParallel->pre_processing();
  rectanglMethodParallel->run();
  rectanglMethodParallel->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  //  Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(rectanglMethodParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

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
