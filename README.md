# GPT.rs

GPT-2 model inference implemented in (nearly) pure Rust.

## Usage

Download the model from [here](gpt2_124M.bin) and move it to the current directory.

```shell
cargo run --release "Ladies and"
```

## Acknowledgements

- [jyy-M3](https://jyywiki.cn/OS/2024/labs/M3.md). The initial code is translated from the C code in this lab.
- [llm.c](https://github.com/karpathy/llm.c)
