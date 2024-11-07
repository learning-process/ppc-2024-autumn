#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "seq/vershinina_a_integration_the_monte_carlo_method/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

std::vector<float> vershinina_a_integration_the_monte_carlo_method::getRandomVector(float sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(0, 100);
  std::vector<float> vec(sz);
  vec[0] = distr(gen);
  vec[1] = vec[0] + distr(gen);
  vec[2] = distr(gen);
  vec[3] = vec[2] + distr(gen);
  return vec;
}

bool vershinina_a_integration_the_monte_carlo_method::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = reinterpret_cast<float*>(taskData->inputs[0]);
  xmin = input_[0];
  xmax = input_[1];
  ymin = input_[2];
  ymax = input_[3];
  return true;
}

bool vershinina_a_integration_the_monte_carlo_method::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1;
}

bool vershinina_a_integration_the_monte_carlo_method::TestTaskSequential::run() {
  internal_order_test();
  int count;
  float total = 0;
  float inBox = 0;
  reference_res = 0;
  for (count = 0; count < 100000; count++) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;

    float xcoord = ((xmax - xmin) * u1) + xmin;
    float ycoord = ((ymax - ymin) * u2) + ymin;
    float val = p(xcoord);

    ++total;

    if (val > ycoord) {
      ++inBox;
    }
  }
  float density = inBox / total;

  reference_res = (xmax - xmin) * (ymax - ymin) * density;
  return true;
}
bool vershinina_a_integration_the_monte_carlo_method::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<float*>(taskData->outputs[0])[0] = reference_res;
  return true;
}
