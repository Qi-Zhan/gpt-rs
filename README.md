# GPT.rs

GPT-2 model inference implemented in (nearly) pure Rust.

## Usage

Download the model from the [NJU GPT-2](https://box.nju.edu.cn/f/da66441d8c6d48d6b848/) and extract it to the current directory.

```shell
cargo run --release "Ladies and"
```

## Acknowledgements

- [jyy-M3](https://jyywiki.cn/OS/2024/labs/M3.md). The initial code is translated from the C code in this lab.
- [llm.c](https://github.com/karpathy/llm.c)
