#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace fomin_v_generalized_scatter {

// Declaration of TaskData struct
struct TaskData {
  void* inputs[1];
  int inputs_count[1];
  void* outputs[1];
  int outputs_count[1];
  MPI_Datatype datatype;
  std::string ops;
};

// Declaration of getRandomVector function
std::vector<int> getRandomVector(int sz);
// Declaration of generalized_scatter function
int generalized_scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                        MPI_Datatype recvtype, int root, MPI_Comm comm);

// Declaration of GeneralizedScatterTestParallel class
class GeneralizedScatterTestParallel : public ppc::core::Task {
 public:
  bool pre_processing(TaskData* taskData) override;
  bool validation(TaskData* taskData) override;
  bool run(TaskData* taskData) override;
  bool post_processing(TaskData* taskData) override;

 private:
};

}  // namespace fomin_v_generalized_scatter
