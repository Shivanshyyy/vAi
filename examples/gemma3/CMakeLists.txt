cmake_minimum_required(VERSION 3.16)

project(demo-gemma3)

# demo-gemma3 — one-prompt completion with timings
add_executable(demo-gemma3
    demo.cpp
)

# chat-gemma3 — multi-turn chat on the KV cache, with an image on 4B
add_executable(chat-gemma3
    chat.cpp
)
