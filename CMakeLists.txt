cmake_minimum_required(VERSION 3.20)
project(cpp_rag_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# Fetch nlohmann/json (header only)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(rag
  src/main.cpp
  src/ollama_client.cpp
  src/minijson.cpp
  src/vector_store.cpp
  src/text_chunker.cpp
  src/io_utils.cpp
)

if (MSVC)
  target_compile_options(rag PRIVATE /W4)
else()
  target_compile_options(rag PRIVATE -Wall -Wextra -Wpedantic)
endif()
