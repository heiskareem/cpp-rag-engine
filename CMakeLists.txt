cmake_minimum_required(VERSION 3.20)
project(cpp_rag_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# Fetch llama.cpp
FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  # Use a moving branch to avoid broken pinned SHAs; pin later if desired
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(llama_cpp)

# Fetch nlohmann/json (header only)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(rag
  src/main.cpp
  src/llama_embedder.cpp
  src/llama_generator.cpp
  src/vector_store.cpp
  src/text_chunker.cpp
  src/io_utils.cpp
)

target_include_directories(rag PRIVATE
  ${llama_cpp_SOURCE_DIR}
  ${llama_cpp_SOURCE_DIR}/include
  ${llama_cpp_SOURCE_DIR}/common
)

# Link llama library target if present
find_library(LLAMA_LIB NAMES llama PATHS ${llama_cpp_BINARY_DIR} ${llama_cpp_SOURCE_DIR} PATH_SUFFIXES lib bin)
if (LLAMA_LIB)
  target_link_libraries(rag PRIVATE ${LLAMA_LIB})
else()
  # Fall back to CMake target name exported by llama.cpp
  target_link_libraries(rag PRIVATE llama)
endif()

target_link_libraries(rag PRIVATE nlohmann_json::nlohmann_json)

if (MSVC)
  target_compile_options(rag PRIVATE /W4)
else()
  target_compile_options(rag PRIVATE -Wall -Wextra -Wpedantic)
endif()
