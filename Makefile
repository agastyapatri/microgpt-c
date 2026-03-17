CC = clang 
CFLAGS = -std=c17 -Wall -Wextra  -g -O0 -march=native 
SRCS = ad.c utils.c main.c 
all: main 

main: $(SRCS)
	$(CC) $(CFLAGS)  $^ -o $@ -lm 
