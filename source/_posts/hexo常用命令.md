---
title: hexo常用命令
date: 2024-12-04 16:43:26
tags: 
- hexo
- 部署
categories: 
- hexo
---

## 新建一篇文章
```bash
hexo new "文章标题"
```

## 本地重新生成静态文件
```bash
hexo clean
hexo g
```

## 发布到github.io
```bash
hexo d
```

## 在文章里面插入图片
* 首先把图片放到source/images下面
* 然后文章里插入：
```markdown
<img src="/images/图片文件名" width="50%">
```