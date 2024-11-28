#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/map.hpp>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sadikov_I_gather_mpi {//test
int DataLocation(int rank, int delta, int maxDelta);
void Gather(const boost::mpi::communicator& world, std::map<int, int> id, std::vector<int>& inValues,
            const std::vector<int>& sizes, std::vector<int>& outValues, int size, int root);
class MPITask : public ppc::core::Task {
 private:
  std::vector<int> m_sum;
  std::vector<int> m_matrix;
  size_t m_rowsCount, m_columnsCount = 0;

 public:
  explicit MPITask(std::shared_ptr<ppc::core::TaskData> td);
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  void Accumulate(size_t size);
};

class MPITaskParallel : public ppc::core::Task {
 private:
  std::vector<int> m_sum;
  std::vector<int> m_matrix;
  std::vector<int> m_localInput;
  size_t m_rowsCount, m_columnsCount = 0;
  size_t m_lastColumn = 0;
  size_t m_delta = 0;
  int m_root = 0;
  std::vector<int> m_sizes;
  std::map<int, int> m_id;
  boost::mpi::communicator world;

 public:
  explicit MPITaskParallel(std::shared_ptr<ppc::core::TaskData> td);
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  std::vector<int> Accumulate(size_t size);
  std::vector<int> CalculateSize(int elements, int count);
  void SetRoot(int root);
};

class ReferenceTask : public ppc::core::Task {
 private:
  std::vector<int> m_sum;
  std::vector<int> m_matrix;
  std::vector<int> m_localInput;
  size_t m_rowsCount, m_columnsCount = 0;
  size_t m_lastColumn = 0;
  size_t m_delta = 0;
  boost::mpi::communicator world;

 public:
  explicit ReferenceTask(std::shared_ptr<ppc::core::TaskData> td);
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  std::vector<int> Accumulate(size_t size);
  std::vector<int> CalculateSize(int elements, int count);
};
}  // namespace sadikov_I_gather_mpi