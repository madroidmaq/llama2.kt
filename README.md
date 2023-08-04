# Llama2.kt

![](docs/llama2.kt.png)

Port of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) to Kotlin.


## Performance

以下数据都是基于 Macbook Pro 2019(2.3 GHz 八核Intel Core i9) 测试。

|          | parameters                                                   | tok/s |
| -------- | ------------------------------------------------------------ | ----- |
| Llama.c  | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) | 26    |
| Llama.c  | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) | 9     |
| Llama.c  | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) | 3     |
| Llama.kt | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) | 61    |
| Llama.kt | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) | 22    |
| Llama.kt | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) | 8     |



> 注意，以上数据使用 `gcc -o run run.c -lm` 进行编译、测试。


## How to RUN

![](docs/model_15M.png)

方式1: 编译 jar 包并执行

```shell
kotlinc src/main/kotlin/Llama2.kt -include-runtime -d Llama2.jar
java -jar Llama2.jar /path/to/model.bin
```

方式2：使用 gradle task

```shell
./gradlew completion -Pcp="/path/to/model.bin"
```





