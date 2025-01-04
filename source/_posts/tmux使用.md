---
title: tmux使用
date: 2024-12-15 20:32:07
tags:
- tmux
categories:
- 实验室实践
---
## 需求
* 我想在服务器上运行一个程序，但是这个程序耗时很长，我想断掉我本地机器和服务器的连接然后关机睡觉，但是我希望服务器上继续在跑
* 解决方案：**tmux**

## tmux安装
* 因为我用的服务器是组里面的多人服务器，所以我不能用root权限直接安装，又懒得用源码，于是就用conda虚拟环境安装
```shell
conda install -c conda-forge tmux
tmux -V
```

## tmux使用
### 创建一个新的tmux会话
```shell
tmux new-session -s mysession
```
* mysession是给这个会话起的名字，可以随便取
* 然后就进入了这个tmux会话，正常输入要跑的程序命令，就开始跑了
### 断开tmux会话（保持程序在后台运行）
* 快捷键：ctrl+B，然后松开，再按D
* 然后就被带回到了原来的shell界面，但是tmux会话还在后台运行
* 这个时候就可以断掉本地机器和服务器的连接了，关机美美睡觉
### 重新连接tmux会话
```shell
tmux attach-session -t mysession
```
* 如果忘记了会话名字，可以用`tmux ls`查看所有会话
### 退出tmux会话
* 在tmux会话中输入`exit`，或者直接关闭终端窗口
