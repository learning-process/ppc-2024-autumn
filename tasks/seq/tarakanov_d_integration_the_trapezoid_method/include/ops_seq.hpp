// Copyright 2024 Tarakanov Denis
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace tarakanov_d_integration_the_trapezoid_method_seq 
{

class integration_the_trapezoid_method : public ppc::core::Task 
{
 public:
  explicit integration_the_trapezoid_method(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double a{}, b{}, h{}, res{};;

  double f(double x) 
  {
    return x * x;
  };
};

}  // namespace tarakanov_d_integration_the_trapezoid_method_seq