#pragma once
#include <mutex>
#include <vector>

namespace konkov_i_task_dining_philosophers {

class DiningPhilosophers {
 public:
  DiningPhilosophers(int philosophers, int meals);
  void run();
  void getResults(std::vector<int>& results);

 private:
  int philosopher_count_;
  int meals_per_philosopher_;
  std::vector<int> meal_counts_;
  std::vector<std::mutex> forks_;
  void philosopherTask(int id);
};
}  // namespace konkov_i_task_dining_philosophers
