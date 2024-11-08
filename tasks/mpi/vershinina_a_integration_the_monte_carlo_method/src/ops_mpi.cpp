#include "mpi/vershinina_a_integration_the_monte_carlo_method/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<double> vershinina_a_integration_the_monte_carlo_method::getRandomVector(double sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(0, 100);
  std::vector<double> vec(sz);
  vec[0] = distr(gen);
  vec[1] = vec[0] + distr(gen);
  vec[2] = distr(gen);
  vec[3] = vec[2] + distr(gen);
  return vec;
}

bool vershinina_a_integration_the_monte_carlo_method::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = reinterpret_cast<double*>(taskData->inputs[0]);
  xmin = input_[0];
  xmax = input_[1];
  ymin = input_[2];
  ymax = input_[3];
  return true;
}

bool vershinina_a_integration_the_monte_carlo_method::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1;
}

bool vershinina_a_integration_the_monte_carlo_method::TestMPITaskSequential::run() {
  internal_order_test();
  int count;
  double total = 0;
  double inBox = 0;
  reference_res = 0;
  for (count = 0; count < 1000000; count++) {
    double u1 = (double)rand() / (double)RAND_MAX;
    double u2 = (double)rand() / (double)RAND_MAX;

    double xcoord = ((xmax - xmin) * u1) + xmin;
    double ycoord = ((ymax - ymin) * u2) + ymin;
    double val = p(xcoord);

    ++total;

    if (val > ycoord) {
      ++inBox;
    }
  }
  double density = inBox / total;

  reference_res = (xmax - xmin) * (ymax - ymin) * density;
  return true;
}
bool vershinina_a_integration_the_monte_carlo_method::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = reference_res;
  return true;
}

bool vershinina_a_integration_the_monte_carlo_method::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* pr = reinterpret_cast<double*>(taskData->inputs[0]);
    input_.resize(taskData->inputs_count[0]);
    std::copy(pr, pr + input_.size(), input_.begin());
  }
  return true;
}

bool vershinina_a_integration_the_monte_carlo_method::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool vershinina_a_integration_the_monte_carlo_method::TestMPITaskParallel::run() {
  internal_order_test();
  size_t input_size = input_.size();
  broadcast(world, input_size, 0);
  input_.resize(input_size);
  broadcast(world, input_.data(), input_size, 0);

  xmin = input_[0];
  xmax = input_[1];
  ymin = input_[2];
  ymax = input_[3];
  global_res = 0;
  int count;
  local_total = 0;
  local_inBox = 0;
  double total = 0;
  double inBox = 0;
  auto tgt = (1000000 / world.size()) * (world.rank() + 1);
  for (count = (1000000 / world.size()) * world.rank(); count < tgt; count++) {
    double u1 = (double)rand() / (double)RAND_MAX;
    double u2 = (double)rand() / (double)RAND_MAX;

    double xcoord = ((xmax - xmin) * u1) + xmin;
    double ycoord = ((ymax - ymin) * u2) + ymin;

    double val = p(xcoord);

    ++local_total;

    if (val > ycoord) {
      ++local_inBox;
    }
  }
  reduce(world, local_total, total, std::plus(), 0);
  reduce(world, local_inBox, inBox, std::plus(), 0);

  double density = inBox / total;
  global_res = (xmax - xmin) * (ymax - ymin) * density;

  return true;
}

bool vershinina_a_integration_the_monte_carlo_method::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = global_res;
  }
  return true;
}
