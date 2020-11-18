# Spark Tinkering

To run Spark + Jupyter container. Then go to [http://localhost:8888](http://localhost:8888).

On Linux.

```bash
docker run -it \
    -p 9870:9870 \
    -p 8088:8088 \
    -p 8080:8080 \
    -p 18080:18080 \
    -p 9000:9000 \
    -p 8888:8888 \
    -p 9864:9864 \
    -v $HOME/git/py-pair/misc/ipynb:/root/ipynb \
    -e PYSPARK_MASTER=spark://localhost:7077 \
    -e NOTEBOOK_PASSWORD='' \
    oneoffcoder/spark-jupyter
```

On Windows.

```bash
docker run -it ^
    -p 9870:9870 ^
    -p 8088:8088 ^
    -p 8080:8080 ^
    -p 18080:18080 ^
    -p 9000:9000 ^
    -p 8888:8888 ^
    -p 9864:9864 ^
    -v ./git/py-pair/misc/ipynb:/root/ipynb ^
    -e PYSPARK_MASTER=spark://localhost:7077 ^
    -e NOTEBOOK_PASSWORD='' ^
    oneoffcoder/spark-jupyter
```