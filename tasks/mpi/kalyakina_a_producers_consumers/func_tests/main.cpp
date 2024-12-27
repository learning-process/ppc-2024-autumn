// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kalyakina_a_producers_consumers/include/ops_mpi.hpp"

TEST(kalyakina_a_producers_consumers_mpi, Test_of_validation_invalid_count_of_input_datas) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5 / 2 / 2, 1), std::max(world.size() / 5, 1),
                          std::max(world.size() / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_EQ(TaskParallel.validation(), false);
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_of_validation_invalid_count_of_output_datas) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5 / 2 / 2, 1), std::max(world.size() / 5, 1),
                          std::max(world.size() / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_EQ(TaskParallel.validation(), false);
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_of_validation_invalid_size_of_buffer) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{0, std::max(world.size() / 5, 1), std::max(world.size() / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_EQ(TaskParallel.validation(), false);
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_of_validation_invalid_count_of_producers_too_few) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5 / 2 / 2, 1), 0, std::max(world.size() / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_EQ(TaskParallel.validation(), false);
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_of_validation_invalid_count_of_producers_too_many) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5 / 2 / 2, 1), world.size(), std::max(world.size() / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_EQ(TaskParallel.validation(), false);
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_20_percent_of_producers_small_buffer_few_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5 / 2 / 2, 1), std::max(world.size() / 5, 1),
                          std::max(world.size() / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_20_percent_of_producers_small_buffer_average_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5 / 4, 1), std::max(world.size() / 5, 1),
                          std::max(world.size() / 5, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_20_percent_of_producers_small_buffer_many_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5 / 4, 1), std::max(world.size() / 5, 1), world.size()};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_20_percent_of_producers_average_buffer_few_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5, 1), std::max(world.size() / 5, 1),
                          std::max(world.size() / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_20_percent_of_producers_average_buffer_average_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5, 1), std::max(world.size() / 5, 1), std::max(world.size() / 5, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_20_percent_of_producers_average_buffer_many_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5, 1), std::max(world.size() / 5, 1), world.size()};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_20_percent_of_producers_big_buffer_few_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 5 / 2, 1), std::max(world.size() / 5, 1),
                          std::max(world.size() / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_20_percent_of_producers_big_buffer_many_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{world.size(), std::max(world.size() / 5, 1), world.size()};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_50_percent_of_producers_small_buffer_few_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 2 / 2 / 2, 1), std::max(world.size() / 2, 1),
                          std::max(world.size() / 2 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_50_percent_of_producers_small_buffer_average_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 2 / 4, 1), std::max(world.size() / 2, 1),
                          std::max(world.size() / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_50_percent_of_producers_small_buffer_many_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 2 / 4, 1), std::max(world.size() / 2, 1), world.size()};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_50_percent_of_producers_average_buffer_few_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 2, 1), std::max(world.size() / 2, 1),
                          std::max(world.size() / 2 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_50_percent_of_producers_average_buffer_average_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 2, 1), std::max(world.size() / 2, 1), std::max(world.size() / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_50_percent_of_producers_average_buffer_many_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 2, 1), std::max(world.size() / 2, 1), world.size()};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_50_percent_of_producers_big_buffer_few_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() / 2 / 2, 1), std::max(world.size() / 2, 1),
                          std::max(world.size() / 2 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_50_percent_of_producers_big_buffer_many_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{world.size(), std::max(world.size() / 2, 1), world.size()};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_80_percent_of_producers_small_buffer_few_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() * 4 / 5 / 2 / 2, 1),
                          std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1),
                          std::max(world.size() * 4 / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_80_percent_of_producers_small_buffer_average_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() * 4 / 5 / 4, 1),
                          std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1),
                          std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_80_percent_of_producers_small_buffer_many_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() * 4 / 5 / 4, 1),
                          std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1), world.size()};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_80_percent_of_producers_average_buffer_few_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1),
                          std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1),
                          std::max(world.size() * 4 / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_80_percent_of_producers_average_buffer_average_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1),
                          std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1),
                          std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_80_percent_of_producers_average_buffer_many_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1),
                          std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1), world.size()};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_80_percent_of_producers_big_buffer_few_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{std::max(world.size() * 4 / 5 / 2, 1),
                          std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1),
                          std::max(world.size() * 4 / 5 / 2, 1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}

TEST(kalyakina_a_producers_consumers_mpi, Test_about_80_percent_of_producers_big_buffer_many_souces) {
  boost::mpi::communicator world;

  std::vector<int> in{};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{world.size(), std::min(std::max(world.size() * 4 / 5, 1), world.size() - 1), world.size()};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel TaskParallel(taskDataPar);
  if (world.size() < 2) {
    if (world.rank() == 0) {
      ASSERT_EQ(TaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(TaskParallel.validation(), true);
    TaskParallel.pre_processing();
    TaskParallel.run();
    TaskParallel.post_processing();

    if (world.rank() == 0) {
      ASSERT_EQ(out[0], 0);
    }
  }
}