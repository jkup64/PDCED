# 环境准备
推荐使用conda：
```bash
$ conda create --name <env> --file requirements.txt
```
或者在已创建的虚拟环境中
```bash
$ conda install --yes --file requirements.txt
```


# 运行
server：
```python
$ python cloud.py
```

client：
```bash
$ python edge.py
```