cmake_minimum_required(VERSION 3.13.4)
set(TARGET "monodomain")
project(${TARGET} CXX)

find_package(deal.II 9.6.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

set(TARGET_SRC ${TARGET}.cpp)
deal_ii_initialize_cached_variables()
deal_ii_invoke_autopilot()
