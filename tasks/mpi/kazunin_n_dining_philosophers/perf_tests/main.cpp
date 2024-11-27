#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"

namespace kazunin_n_dining_philosophers {

class DiningPhilosophersPerformanceTest : public ::testing::Test {
 protected:
  boost::mpi::communicator mpi_comm;
  int philosophers_count;

  void SetUp() override { philosophers_count = mpi_comm.size(); }

  std::shared_ptr<ppc::core::TaskData> create_task_data() {
    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs_count.emplace_back(philosophers_count);
    return task_data;
  }

  template <typename TaskType>
  void validate_and_execute(const std::shared_ptr<TaskType>& task) {
    ASSERT_TRUE(task->validation()) << "Validation failed for philosophers task!";

    task->pre_processing();
    task->run();
    task->post_processing();
  }

  void analyze_performance(const std::shared_ptr<ppc::core::Perf>& analyzer,
                           const std::shared_ptr<ppc::core::PerfAttr>& attributes,
                           const std::shared_ptr<ppc::core::PerfResults>& results) {
    analyzer->pipeline_run(attributes, results);
    if (mpi_comm.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(results);
    }
  }

  void analyze_task_performance(const std::shared_ptr<ppc::core::Perf>& analyzer,
                                const std::shared_ptr<ppc::core::PerfAttr>& attributes,
                                const std::shared_ptr<ppc::core::PerfResults>& results) {
    analyzer->task_run(attributes, results);
    if (mpi_comm.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(results);
    }
  }
};

TEST_F(DiningPhilosophersPerformanceTest, PipelineExecution) {
  if (philosophers_count < 2) {
    GTEST_SKIP() << "Not enough philosophers for meaningful execution.";
  }

  auto task_data = create_task_data();
  auto philosophers_task = std::make_shared<KazuninDiningPhilosophersMPI<int>>(task_data);

  validate_and_execute(philosophers_task);

  auto performance_attr = std::make_shared<ppc::core::PerfAttr>();
  performance_attr->num_running = 20;
  const boost::mpi::timer execution_timer;
  performance_attr->current_timer = [&] { return execution_timer.elapsed(); };

  auto performance_results = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(philosophers_task);

  analyze_performance(analyzer, performance_attr, performance_results);

  if (mpi_comm.rank() == 0) {
    ASSERT_FALSE(philosophers_task->detect_deadlock()) << "Deadlock detected during pipeline execution!";
  }
}

TEST_F(DiningPhilosophersPerformanceTest, TaskExecution) {
  if (philosophers_count < 2) {
    GTEST_SKIP() << "Not enough philosophers for meaningful execution.";
  }

  auto task_data = create_task_data();
  auto philosophers_task = std::make_shared<KazuninDiningPhilosophersMPI<int>>(task_data);

  validate_and_execute(philosophers_task);

  auto performance_attr = std::make_shared<ppc::core::PerfAttr>();
  performance_attr->num_running = 10;
  const boost::mpi::timer execution_timer;
  performance_attr->current_timer = [&] { return execution_timer.elapsed(); };

  auto performance_results = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(philosophers_task);

  analyze_task_performance(analyzer, performance_attr, performance_results);

  if (mpi_comm.rank() == 0) {
    ASSERT_FALSE(philosophers_task->detect_deadlock()) << "Deadlock detected during task execution!";
  }
}

}  // namespace kazunin_n_dining_philosophers
