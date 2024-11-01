#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
namespace sadikov_I_Sum_values_by_columns_matrix_mpi {
std::shared_ptr<ppc::core::TaskData> CreateTaskData(std::vector<int> &InV, std::vector<int> &CeV,
                                                    std::vector<int> &OtV);
std::vector<int> getRandomVector(size_t size);
class MPITask : public ppc::core::Task {
 private:
  std::vector<int> sum;
  std::vector<int> matrix;
  size_t rows_count, columns_count = 0;

 public:
  explicit MPITask(std::shared_ptr<ppc::core::TaskData> td);
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  void calculate(size_t size);
};

class MPITaskParallel : public ppc::core::Task {
 private:
  std::vector<int> sum;
  std::vector<int> matrix;
  std::vector<int> local_input;
  size_t rows_count, columns_count = 0;
  size_t last_column = 0;
  size_t delta = 0;
  boost::mpi::communicator world;

 public:
  explicit MPITaskParallel(std::shared_ptr<ppc::core::TaskData> td);
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  std::vector<int> calculate(size_t size);
};
}  // namespace sadikov_I_Sum_values_by_columns_matrix_mpi