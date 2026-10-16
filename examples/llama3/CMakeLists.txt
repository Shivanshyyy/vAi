cmake_minimum_required(VERSION 3.16)

project(demo-llama3)

# demo-llama3 — one-prompt completion with timings
add_executable(demo-llama3
    demo.cpp
)

# chat-llama3 — multi-turn interactive chat on the KV cache
add_executable(chat-llama3
    chat.cpp
)
