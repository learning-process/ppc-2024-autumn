#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chistov_a_gather_my {

template <typename T>
std::vector<T> getRandomVector(int size_) {
  if (size_ < 0) {
    return std::vector<T>();
  }

  std::vector<T> randomVector(size_);
  for (int i = 0; i < size_; ++i) {
    randomVector[i] = static_cast<T>((std::rand() % 201) - 100);
  }

  return randomVector;
}

template <typename T>
class Gather : public ppc::core::Task {
 public:
  explicit Gather(std::shared_ptr<ppc::core::TaskData> taskData_, int root_)
      : Task(std::move(taskData_)), root(root_) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> sendbuf;
  int root{};
  int leftChild{};
  int rightChild{};
  int parent{};
  boost::mpi::communicator world;
};

template <typename T>
bool gather(const boost::mpi::communicator& world, std::vector<T>& local_vector, int count,
            std::vector<T>& gathered_data, int root) {
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vector.data()));
  taskDataPar->inputs_count.emplace_back(count);

  if (world.rank() == root) {
    gathered_data.resize(world.size() * count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_data.data()));
  }

  Gather<T> gatherTask(taskDataPar, root);
  if (!gatherTask.validation()) {
    return false;
  }

  gatherTask.pre_processing();
  gatherTask.run();
  gatherTask.post_processing();

  return true;
}

}  // namespace chistov_a_gather_my