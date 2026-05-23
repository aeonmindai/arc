//! C++ trajectories (5).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        cpp_021_unique_ptr_refactor(),
        cpp_022_string_view(),
        cpp_023_thread_pool(),
        cpp_024_concept_constraint(),
        cpp_025_cmake_target(),
    ]
}

fn cpp_021_unique_ptr_refactor() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "src/cache.cpp uses raw new/delete for Node. Refactor to std::unique_ptr<Node> and remove the destructor's manual cleanup.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"src/cache.cpp"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "#include \"cache.h\"\n\nCache::Cache() : head(new Node()) {}\nCache::~Cache() { delete head; head = nullptr; }\n\nvoid Cache::Insert(int key) {\n    Node* n = new Node{key, head->next};\n    head->next = n;\n}\n",
    ));
    turns.push(assistant_call(
        "Reading header to see Node layout.",
        "call_002",
        "read_file",
        json!({"path":"src/cache.h"}),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "#pragma once\nstruct Node { int key; Node* next = nullptr; };\nclass Cache {\npublic:\n    Cache();\n    ~Cache();\n    void Insert(int key);\nprivate:\n    Node* head;\n};\n",
    ));
    turns.push(assistant_call(
        "Updating the header to use unique_ptr and dropping the destructor.",
        "call_003",
        "write_file",
        json!({
            "path":"src/cache.h",
            "contents":"#pragma once\n#include <memory>\nstruct Node { int key; std::unique_ptr<Node> next; };\nclass Cache {\npublic:\n    Cache();\n    void Insert(int key);\nprivate:\n    std::unique_ptr<Node> head;\n};\n"
        }),
        &turns,
    ));
    turns.push(tool("call_003", "wrote 252 bytes to src/cache.h"));
    turns.push(assistant_call(
        "Updating the cpp.",
        "call_004",
        "write_file",
        json!({
            "path":"src/cache.cpp",
            "contents":"#include \"cache.h\"\n\nCache::Cache() : head(std::make_unique<Node>()) {}\n\nvoid Cache::Insert(int key) {\n    auto n = std::make_unique<Node>();\n    n->key = key;\n    n->next = std::move(head->next);\n    head->next = std::move(n);\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_004", "wrote 252 bytes to src/cache.cpp"));
    turns.push(assistant_text(
        "Cache now uses unique_ptr throughout; the manual destructor and `new` calls are gone.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_021",
        "cpp",
        src("github.com/abseil/abseil-cpp", "Apache-2.0", "20240722"),
        turns,
    )
}

fn cpp_022_string_view() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Refactor Trim() in str_util.cpp to accept std::string_view and return std::string_view (no allocation).",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"str_util.cpp"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "#include \"str_util.h\"\n#include <cctype>\n\nstd::string Trim(const std::string& s) {\n    size_t i = 0; while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;\n    size_t j = s.size(); while (j > i && std::isspace((unsigned char)s[j-1])) --j;\n    return s.substr(i, j - i);\n}\n",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_002",
        "write_file",
        json!({
            "path":"str_util.cpp",
            "contents":"#include \"str_util.h\"\n#include <cctype>\n\nstd::string_view Trim(std::string_view s) {\n    size_t i = 0; while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;\n    size_t j = s.size(); while (j > i && std::isspace(static_cast<unsigned char>(s[j-1]))) --j;\n    return s.substr(i, j - i);\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 388 bytes to str_util.cpp"));
    turns.push(assistant_text(
        "Trim now operates on string_view with zero allocation; callers can wrap the result in std::string only if they need ownership.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_022",
        "cpp",
        src("github.com/google/googletest", "BSD-3-Clause", "5e2c0a8"),
        turns,
    )
}

fn cpp_023_thread_pool() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a header-only ThreadPool in include/thread_pool.h. Constructor takes worker count; submit(F&& f) returns std::future<void>. Use std::jthread.",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_001",
        "write_file",
        json!({
            "path":"include/thread_pool.h",
            "contents":"#pragma once\n#include <condition_variable>\n#include <functional>\n#include <future>\n#include <memory>\n#include <mutex>\n#include <queue>\n#include <stop_token>\n#include <thread>\n#include <vector>\n\nclass ThreadPool {\npublic:\n    explicit ThreadPool(std::size_t n) {\n        workers_.reserve(n);\n        for (std::size_t i = 0; i < n; ++i) {\n            workers_.emplace_back([this](std::stop_token st) { worker(st); });\n        }\n    }\n    ~ThreadPool() {\n        for (auto& w : workers_) w.request_stop();\n        cv_.notify_all();\n    }\n    std::future<void> submit(std::function<void()> f) {\n        auto task = std::make_shared<std::packaged_task<void()>>(std::move(f));\n        auto fut = task->get_future();\n        {\n            std::lock_guard lk(mu_);\n            queue_.emplace([task] { (*task)(); });\n        }\n        cv_.notify_one();\n        return fut;\n    }\nprivate:\n    void worker(std::stop_token st) {\n        while (!st.stop_requested()) {\n            std::function<void()> job;\n            {\n                std::unique_lock lk(mu_);\n                cv_.wait(lk, [&] { return !queue_.empty() || st.stop_requested(); });\n                if (st.stop_requested() && queue_.empty()) return;\n                job = std::move(queue_.front());\n                queue_.pop();\n            }\n            job();\n        }\n    }\n    std::vector<std::jthread> workers_;\n    std::mutex mu_;\n    std::condition_variable cv_;\n    std::queue<std::function<void()>> queue_;\n};\n"
        }),
        &turns,
    ));
    turns.push(tool("call_001", "wrote 1408 bytes to include/thread_pool.h"));
    turns.push(assistant_text(
        "Header-only thread pool ready. Workers are std::jthread; the destructor requests stop and the condition variable wakes idle workers so they observe the stop token.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_023",
        "cpp",
        src("github.com/progschj/ThreadPool", "Zlib", "9a42ec1"),
        turns,
    )
}

fn cpp_024_concept_constraint() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a C++20 concept Hashable<T> and constrain MakeIndex<T> in include/index.h to require it.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"include/index.h"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "#pragma once\n#include <unordered_set>\n\ntemplate <class T>\nstd::unordered_set<T> MakeIndex(const std::vector<T>& items) {\n    return std::unordered_set<T>(items.begin(), items.end());\n}\n",
    ));
    turns.push(assistant_call(
        "Adding the concept.",
        "call_002",
        "write_file",
        json!({
            "path":"include/index.h",
            "contents":"#pragma once\n#include <concepts>\n#include <functional>\n#include <unordered_set>\n#include <vector>\n\ntemplate <class T>\nconcept Hashable = requires(const T& v) {\n    { std::hash<T>{}(v) } -> std::convertible_to<std::size_t>;\n};\n\ntemplate <Hashable T>\nstd::unordered_set<T> MakeIndex(const std::vector<T>& items) {\n    return std::unordered_set<T>(items.begin(), items.end());\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 426 bytes to include/index.h"));
    turns.push(assistant_text(
        "MakeIndex now requires Hashable; the compiler will emit a precise diagnostic instead of a deep template error for non-hashable T.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_024",
        "cpp",
        src("github.com/llvm/llvm-project", "Apache-2.0", "1b3c5e2"),
        turns,
    )
}

fn cpp_025_cmake_target() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a CMake target `tests` that builds tests/unit_test.cpp, links GTest::gtest_main, and is registered with add_test for CTest.",
    ));
    turns.push(assistant_call(
        "Reading the current root CMakeLists.",
        "call_001",
        "read_file",
        json!({"path":"CMakeLists.txt"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "cmake_minimum_required(VERSION 3.20)\nproject(demo CXX)\nset(CMAKE_CXX_STANDARD 20)\n\nadd_library(demo_lib src/lib.cpp)\ntarget_include_directories(demo_lib PUBLIC include)\n",
    ));
    turns.push(assistant_call(
        "Appending the test target.",
        "call_002",
        "write_file",
        json!({
            "path":"CMakeLists.txt",
            "contents":"cmake_minimum_required(VERSION 3.20)\nproject(demo CXX)\nset(CMAKE_CXX_STANDARD 20)\n\nadd_library(demo_lib src/lib.cpp)\ntarget_include_directories(demo_lib PUBLIC include)\n\nenable_testing()\nfind_package(GTest CONFIG REQUIRED)\nadd_executable(tests tests/unit_test.cpp)\ntarget_link_libraries(tests PRIVATE demo_lib GTest::gtest_main)\nadd_test(NAME demo_tests COMMAND tests)\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 442 bytes to CMakeLists.txt"));
    turns.push(assistant_text(
        "`tests` target is registered with CTest. Run `ctest --test-dir build` after configuring.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_025",
        "cpp",
        src("github.com/Kitware/CMake", "BSD-3-Clause", "2c7e85a"),
        turns,
    )
}
