# 第一版

使用[Qwen3-VL-2B-Instruct模型](https://www.modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct)进行内容生成，适配linux与windows系统。

环境依赖：
- pytorch
- flask

下载好模型后，直接放在当前目录中。

测试时使用miniconda环境，具体配置参照`environment.yml`文件。

本地测试时需要修改`/static/js/main.js`下第一行的`API_BASE`参数为自己的ip地址（局域网地址或127.0.0.1）；部署时需要将ip地址更换为对应域名。程序运行在4543端口。
