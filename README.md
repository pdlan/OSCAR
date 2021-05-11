# OSCAR
This repository contains the source code of our ICML 2021 paper [How could Neural Networks understand Programs?](https://arxiv.org/pdf/2105.04297).
![Architecture](assets/Architecture.png)
## Environment
Run following commands to build a docker image for the environment:

```shell
cd docker
sudo docker build -t oscar:latest .
```

And you can launch a container with `nvidia-docker` command.

```shell
sudo nvidia-docker run -it --mount type=bind,source="$(pwd)",target=/oscar oscar:latest
```

To compile the binaries for processing the data:

```shell
cd /oscar/bin
make
```

Then the OSCAR LLVM analyzer pass (located in `analyzer`), IR Lexer (located in `irlexer`), and FastBPE (located in fastBPE) will be compiled.

## Processing the data

First, please visit [https://1drv.ms/u/s!AjYwgux2zLgMiAhYpoCU3jLu20Z6?e=XR52y9](https://1drv.ms/u/s!AjYwgux2zLgMiAhYpoCU3jLu20Z6?e=XR52y9) to download the data for pretraining and downstream tasks. Extract the downloaded tarballs to the `data-raw` directory.

To process the data for pretraining and the downstream tasks, enter the coressponding directories and execute `./process.sh`. Raw data needs to be placed in the directory `data-raw`. Processed data will be placed in the directory `data-bin`.

## Train the model

Use following commands to pretrain the model:

```shell
cd /oscar/model
./scripts/pretrain.sh
```

For downstream tasks the procedure is similar.