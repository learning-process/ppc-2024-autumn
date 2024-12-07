#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <iostream>
#include <thread>

namespace konkov_i_task_dining_philosophers {

DiningPhilosophers::DiningPhilosophers(int philosophers, int meals)
    : philosopher_count_(philosophers),
      meals_per_philosopher_(meals),
      meal_counts_(philosophers, 0),
      forks_(philosophers),
      world_() {}

void DiningPhilosophers::philosopherTask(int id) {
  for (int i = 0; i < meals_per_philosopher_; ++i) {
    std::lock(forks_[id], forks_[(id + 1) % philosopher_count_]);
    std::lock_guard<std::mutex> left_lock(forks_[id], std::adopt_lock);
    std::lock_guard<std::mutex> right_lock(forks_[(id + 1) % philosopher_count_], std::adopt_lock);

    meal_counts_[id]++;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void DiningPhilosophers::run() {
  std::vector<std::thread> threads;

  for (int i = 0; i < philosopher_count_; ++i) {
    threads.emplace_back(&DiningPhilosophers::philosopherTask, this, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  std::vector<int> global_meal_counts(philosopher_count_, 0);

  boost::mpi::all_reduce(world_, meal_counts_.data(), philosopher_count_, global_meal_counts.data(), std::plus<int>());

  meal_counts_ = global_meal_counts;

  if (world_.rank() == 0) {
    std::cout << "Global meal counts: ";
    for (int count : meal_counts_) {
      std::cout << count << " ";
    }
    std::cout << std::endl;
  }
}

void DiningPhilosophers::getResults(std::vector<int>& results) { results = meal_counts_; }

}  // namespace konkov_i_task_dining_philosophers
