// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/fyodorov_m_num_of_orderly_violations/include/ops_mpi.hpp"

/*
TEST(Parallel_Operations_MPI, test_pipelineaaaaaa_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 120;
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }
  auto testMpiTaskParallel =
std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar, "count");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_violations[0]);
  }
}

TEST(Parallel_Operations_MPI, test_pipeline_run) {
  boost::mpi::environment env; // Initialize MPI
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 120; // Увеличиваем размер вектора для теста
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  auto testMpiTaskParallel =
std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar, "count");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    //std::cout << "Expected: " << count_size_vector << ", Actual: " << global_violations[0] << std::endl;
    ASSERT_EQ(count_size_vector, global_violations[0]);
  }
}



/*
TEST(mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;

  if (world.rank() == 0) {
    count_size_vector = 120; // Большой размер вектора для тестирования производительности
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  auto testMpiTaskParallel =
std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar, "count");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_violations[0]);
  }
}
*/
/*
TEST(mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;

  if (world.rank() == 0) {
    count_size_vector = 120; // Большой размер вектора для тестирования производительности
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  auto testMpiTaskParallel =
std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar, "count");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_violations[0]);
  }
}
*/
/*
TEST(Parallel_Operations_MPI, test_pipelines_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 120;
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  auto testMpiTaskParallel =
std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar, "count");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Create data for sequential test
    std::vector<int32_t> reference_violations(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_violations.data()));
    taskDataSeq->outputs_count.emplace_back(reference_violations.size());

    // Create Task for sequential execution
    fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "count");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_violations[0], global_violations[0]);
  }
}
*/
/*
TEST(Parallel_Operations_MPI, test_pipelineaaaaaasasass_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 1200;
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  auto testMpiTaskParallel =
std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar, "count");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_violations[0]);
  }
}
*/

TEST(Parallel_Operations_MPI, test_pipelinea_run) {
  // Инициализация MPI
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 12000;
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar, "count");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    // Проверка результата
    int expected_violations = 0;
    for (size_t i = 1; i < global_vec.size(); ++i) {
      if (global_vec[i] < global_vec[i - 1]) {
        ++expected_violations;
      }
    }
    ASSERT_EQ(global_violations[0], expected_violations);
  }
}

TEST(mpi_perf_test, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec;
  std::vector<int32_t> global_violations(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 120;
    global_vec = fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar, "count");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);

  auto start_time = std::chrono::high_resolution_clock::now();

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  world.barrier();

  if (world.rank() == 0) {
    int expected_violations = 0;
    for (size_t i = 1; i < global_vec.size(); ++i) {
      if (global_vec[i] < global_vec[i - 1]) {
        ++expected_violations;
      }
    }
    ASSERT_EQ(global_violations[0], expected_violations);

    // Вывод времени выполнения
    std::cout << "Execution time: " << duration << " ms" << std::endl;
  }
}
/*
int main(int argc, char** argv) {
  // Инициализация MPI
  boost::mpi::environment env(argc, argv);

  // Запуск тестов
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();

  // Синхронизация всех процессов перед завершением
  boost::mpi::communicator world;
  world.barrier();

  return result;
}
*/