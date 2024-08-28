基础环境配置

conda：

创建新环境：

```
conda create --name myenv python=3.10
```

这将创建一个名为 `myenv` 的环境，并在其中安装 `Python 3.10`、

删除环境：

```
conda remove --name myenv --all
```

激活环境：

```
conda activate myenv
```

退出环境：

```
conda deactivate myenv
```

导入导出环境：

```
conda env export > environment.yml  
```

（当我们想要保存某个环境的配置信息，例如下载源信息、环境的Python版本信息、安装的包的版本信息等，可以先切换到指定环境下，通过以下命令将这些配置信息导出：）

该命令会将当前的环境配置信息导出到为 environment.yml 配置信息文件，文件导出地址默认为当前工作目录（也可以指定输出的绝对地址）。这时候，我们将配置文件传到另一台电脑，想基于该配置文件创建 conda 环境，可以通过以下命令：

```
conda env create -f environment.yml
```