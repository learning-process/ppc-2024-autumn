#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <random>

namespace malyshev_v_monte_carlo_integration {

bool TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 3 && taskData->outputs.size() == 1);
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
  num_samples = static_cast<int>(1.0 / epsilon);
  return true;
  
}
  bool TestMPITaskSequential::run() {
  internal_order_test();
  double sum = 0.0;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<> dist(a, b);
   for (int i = 0; i < num_samples; ++i) {
    double x = dist(rng);
    sum += function_square(x);
    
  }
   res = (b - a) * sum / num_samples;
  return true;
  
}
 bool TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
  
}
+  bool TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if ((taskData->inputs.size() != 3) || (taskData->outputs.size() != 1)) {
      return false;
      
    }
    double epsilon = 1e-7;
    if (epsilon <= 0) {
      return false;
      
    }
    
  }
  +return true;
  
}
  bool TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
        a = *reinterpret_cast<double*>(taskData->inputs[0]);
        b = *reinterpret_cast<double*>(taskData->inputs[1]);
        epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
        num_samples = static_cast<int>(1.0 / epsilon);
    }

  boost::mpi::broadcast(world, a, 0);
  boost::mpi::broadcast(world, b, 0);
  boost::mpi::broadcast(world, num_samples, 0);
  local_num_samples = num_samples / world.size();
  return true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();
  double local_sum = 0.0;
  std::mt19937 rng(world.rank());
  std::uniform_real_distribution<> dist(a, b);

for (int i = 0; i < local_num_samples; ++i) {
    double x = dist(rng);
    local_sum += function_square(x);
    
  }

  local_sum *= (b - a) / num_samples;
  boost::mpi::reduce(world, local_sum, res, std::plus<>(), 0);
  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  }
  return true;
}

}  // namespace malyshev_v_monte_carlo_integration
