cmake_minimum_required(VERSION 3.13.4)
set(TARGET "monodomain")
project(${TARGET} CXX)

find_package(deal.II 9.6.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR} /opt/dealii/v9.6.0/lib/cmake/deal.II/
)

set(TARGET_SRC ${TARGET}.cpp applied_current.cpp ionic.cpp utils.cpp)

deal_ii_initialize_cached_variables()
deal_ii_invoke_autopilot()
