#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/fomin_v_generalized_scatter/include/ops_mpi.hpp"

TEST(fomin_v_generalized_scatter_mpi, test_pipeline_run_2000) {
    const int count = 2000; 
    boost::mpi::communicator world;
    std::vector<int> out;
    std::vector<int> in;

    
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        in = fomin_v_generalized_scatter::getRandomVector(count); 
        out = std::vector<int>(count / world.size(), 0);
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
        taskData->inputs_count.emplace_back(count);
        taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
        taskData->outputs_count.emplace_back(count / world.size());
    }

    
    auto scatterParallel = std::make_shared<fomin_v_generalized_scatter::GeneralizedScatterTestParallel>(taskData, world);
    ASSERT_EQ(scatterParallel->validation(), true);
    scatterParallel->pre_processing(); 
    scatterParallel->run(); 
    scatterParallel->post_processing(); 

    
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10; 
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

   
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(scatterParallel);
    perfAnalyzer->pipeline_run(perfAttr, perfResults); 

    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults); 
    }
}


TEST(fomin_v_generalized_scatter_mpi, test_task_run_2000) {
    const int count = 2000; 
    boost::mpi::communicator world;
    std::vector<int> out;
    std::vector<int> in;

    
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        in = fomin_v_generalized_scatter::getRandomVector(count); 
        out = std::vector<int>(count / world.size(), 0);
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
        taskData->inputs_count.emplace_back(count);
        taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
        taskData->outputs_count.emplace_back(count / world.size());
    }

    
    auto scatterParallel = std::make_shared<fomin_v_generalized_scatter::GeneralizedScatterTestParallel>(taskData, world);
    ASSERT_EQ(scatterParallel->validation(), true); 
    scatterParallel->pre_processing(); 
    scatterParallel->run(); 
    scatterParallel->post_processing(); 

    
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 30; 
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(scatterParallel);
    perfAnalyzer->task_run(perfAttr, perfResults); 

    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults); 
    }
}