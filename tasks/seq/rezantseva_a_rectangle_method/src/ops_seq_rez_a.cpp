#include "seq/rezantseva_a_rectangle_method/include/ops_seq_rez_a.hpp"

bool rezantseva_a_rectangle_method_seq::RectangleMethodSequential::check_integration_bounds(
    std::vector<std::pair<double, double>>* ib) {
  if (ib == nullptr) {
    std::cerr << "Error: bounds pointer is null." << std::endl;
    return false;
  }

  for (const auto& bound : *ib) {
    if (bound.first >= bound.second) {
      std::cout << "false";
      return false;
    }
  }
  return true;
}

bool rezantseva_a_rectangle_method_seq::RectangleMethodSequential::validation() {
  internal_order_test();

  auto* bounds = reinterpret_cast<std::vector<std::pair<double, double>>*>(taskData->inputs[0]);
  return (taskData->inputs.size() == 2 && taskData->outputs_count[0] == 1 &&
          (taskData->inputs_count[0] == taskData->inputs_count[1]) && check_integration_bounds(bounds));
}

bool rezantseva_a_rectangle_method_seq::RectangleMethodSequential::pre_processing() {
  internal_order_test();

  auto* bounds = reinterpret_cast<std::vector<std::pair<double, double>>*>(taskData->inputs[0]);
  integration_bounds_.assign(bounds->begin(), bounds->end());

  auto* distribution_ptr = reinterpret_cast<std::vector<int>*>(taskData->inputs[1]);
  distribution_.assign(distribution_ptr->begin(), distribution_ptr->end());

  result_ = 0.0;

  return true;
}

bool rezantseva_a_rectangle_method_seq::RectangleMethodSequential::run() {
  internal_order_test();
  int dimension = distribution_.size();
  std::vector<double> widths(dimension);
  int64_t rectangles = 1;

  for (int i = 0; i < dimension; i++) {
    widths[i] = (integration_bounds_[i].second - integration_bounds_[i].first) / static_cast<double>(distribution_[i]);
    rectangles *= distribution_[i];
  }
  std::vector<double> params(dimension);

  for (int i = 0; i < rectangles; i++) {
    int x = i;
    for (int j = 0; j < dimension; j++) {
      int idDimention = x % distribution_[j];
      params[j] = integration_bounds_[j].first + idDimention * widths[j] + widths[j] / 2;
      x /= distribution_[j];
    }
    result_ += func_(params);
  }
  for (int i = 0; i < dimension; i++) {
    result_ *= widths[i];
  }
  return true;
}

bool rezantseva_a_rectangle_method_seq::RectangleMethodSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  return true;
}
