---
title: Makefile一个使用纠错记录
date: 2024-12-11 20:48:43
tags:
- Makefile
categories:
- 实验室实践
---
* 今天写课设的Makefile，遇到了一个问题，记录一下。
```Makefile
# 包含路径设置
INCLUDES := -IC:/source/ignisos/inc -IC:/source/ignisos/inc -I./user/

# 源文件和目标文件
SRCS := user/syscall_lib.c user/syscall_wrap.S init/main.c
OBJS := $(SRCS:.c=.o) $(SRCS:.S=.o)
OFiles := user/*.o \
		init/*.o 
# 输出目录和 ELF 文件
OUTPUT_DIR := elf
OUTPUT_ELF := $(OUTPUT_DIR)/user_program.elf

# 编译规则
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -std=gnu11 -c -fno-builtin -o $@ $<

%.o: %.S
	$(CC) $(CFLAGS) $(INCLUDES) -c -fno-builtin -o $@ $<

# 默认目标
all: $(OUTPUT_ELF)

# 链接规则
$(OUTPUT_ELF): $(OBJS)
	@mkdir -p $(OUTPUT_DIR)
	$(LD) $(LDFLAGS) -EL -msoft-float -march=m14kc -flto -nostartfiles -nostdlib -static -T user.lds -o $@ $(OFiles)
	$(OC) --remove-section .MIPS.abiflags --remove-section .reginfo $@
	$(SZ) $@


# 清理目标
clean:
	rm -f user/*.o init/*.o $(OUTPUT_DIR)/*

.PHONY: all clean

include include.mk
```
* 一直在链接的时候出错，老是报类似这样的错误：
```shell
init/main.c:1:17: fatal error: lib.h: No such file or directory
 #include "lib.h"
                 ^
compilation terminated.
make: *** [init/main.o] Error 1
```
* 然后拿给我对象看，聪明的他发现了问题，在make打印出来的编译命令中，最后链接的这句长这样：
```shell
mips-mti-elf-gcc  -nostdlib -static -T user.lds -o elf/user_program.elf user/syscall_lib.o user/syscall_wrap.S init/main.o user/syscall_lib.c user/syscall_wrap.o init/main.c
```
* 我们用的mips套件，里面的链接器用不了，所以只能用gcc
* 但这里面应该只链接.o文件才对，这句命令里面有.c和.S文件，gcc碰到这几个文件就会开心的跑去编译，结果编译的时候得找头文件啊，我们看看指定头文件路径的编译命令：
```shell
mips-mti-elf-gcc -EL -g -march=m14kc -msoft-float -O1 -G0 -IC:/source/ignisos/inc -IC:/source/ignisos/inc -I./user/ -c -fno-builtin -o user/syscall_wrap.o user/syscall_wrap.S
```
* 这里面-I指定了头文件路径，但是我们看上面那个链接的命令里面，是没有的，于是gcc跑去编译的时候，找不到头文件，就报错了
