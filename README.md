# GPT.rs

GPT-2 model inference implemented in (nearly) pure Rust.

## Usage

Download the model from [gpt2_124M.bin](https://github.com/Qi-Zhan/gpt.rs/releases/download/1.0/gpt2_124M.bin) and move it to the current directory.

```shell
cargo run --release "Ladies and"
```

## Acknowledgements

- [jyy-M3](https://jyywiki.cn/OS/2024/labs/M3.md). The initial code is translated from the C code in this lab.
- [llm.c](https://github.com/karpathy/llm.c)
