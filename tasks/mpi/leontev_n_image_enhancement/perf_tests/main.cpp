#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/leontev_n_image_enhancement/include/ops_mpi.hpp"

namespace leontev_n_image_enhancement_mpi {
std::vector<int> getRandomImage(int sz) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace leontev_n_image_enhancement_mpi

inline void taskEmplacement(std::shared_ptr<ppc::core::TaskData> &taskDataPar, std::vector<int> &global_vec,
                            std::vector<int> &global_sum) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
  taskDataPar->outputs_count.emplace_back(global_sum.size());
}

TEST(leontev_n_image_enhancement_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int width = 2048;
  const int height = 2048;

  std::vector<int> in_vec;
  const int vector_size = width * height * 3;
  std::vector<int> out_vec_par(vector_size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in_vec = leontev_n_image_enhancement_mpi::getRandomImage(vector_size);
    taskEmplacement(taskDataPar, in_vec, out_vec_par);
  }

  auto MPIImgEnhancementParallel =
      std::make_shared<leontev_n_image_enhancement_mpi::MPIImgEnhancementParallel>(taskDataPar);
  ASSERT_EQ(MPIImgEnhancementParallel->validation(), true);
  MPIImgEnhancementParallel->pre_processing();
  MPIImgEnhancementParallel->run();
  MPIImgEnhancementParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 15;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIImgEnhancementParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> out_vec_seq(vector_size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskEmplacement(taskDataSeq, in_vec, out_vec_seq);

    // Create Task
    leontev_n_image_enhancement_mpi::MPIImgEnhancementSequential MPIImgEnhancementSequential(taskDataSeq);
    ASSERT_EQ(MPIImgEnhancementSequential.validation(), true);
    MPIImgEnhancementSequential.pre_processing();
    MPIImgEnhancementSequential.run();
    MPIImgEnhancementSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(leontev_n_image_enhancement_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int width = 2048;
  const int height = 2048;

  std::vector<int> in_vec;
  const int count_size_vector = width * height * 3;
  std::vector<int> out_vec_par(count_size_vector, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in_vec = leontev_n_image_enhancement_mpi::getRandomImage(count_size_vector);
    taskEmplacement(taskDataPar, in_vec, out_vec_par);
  }

  auto MPIImgEnhancementParallel =
      std::make_shared<leontev_n_image_enhancement_mpi::MPIImgEnhancementParallel>(taskDataPar);
  ASSERT_EQ(MPIImgEnhancementParallel->validation(), true);
  MPIImgEnhancementParallel->pre_processing();
  MPIImgEnhancementParallel->run();
  MPIImgEnhancementParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 15;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIImgEnhancementParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> out_vec_seq(count_size_vector, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskEmplacement(taskDataSeq, in_vec, out_vec_seq);

    // Create Task
    leontev_n_image_enhancement_mpi::MPIImgEnhancementSequential MPIImgEnhancementSequential(taskDataSeq);
    ASSERT_EQ(MPIImgEnhancementSequential.validation(), true);
    MPIImgEnhancementSequential.pre_processing();
    MPIImgEnhancementSequential.run();
    MPIImgEnhancementSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}