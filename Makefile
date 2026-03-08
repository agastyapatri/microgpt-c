CC = clang 
CFLAGS = -std=c17 -Wall -Wextra -Wpedantic -g -O0 
SRCS = ad.c utils.c microgpt.c 
all: main 

main: $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@ -lm 
