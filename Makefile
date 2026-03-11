CC = clang 
CFLAGS = -std=c17 -Wall -Wextra -Wpedantic -g -O0 -march=native 
SRCS = matrix.c autograd.c utils.c microgpt.c 
all: main 

main: $(SRCS)
	$(CC) $(CFLAGS)  $^ -o $@ -lm 
