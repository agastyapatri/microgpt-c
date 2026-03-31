CC = clang 
CFLAGSRELEASE = -std=c17 -Wall -Wextra -O3 -ffast-math -DNDEBUG -march=native 
CFLAGSDEBUG = -std=c17 -Wall -Wextra -O0  -DDEBUG -march=native 

SRCS = ad.c utils.c main.c 
all: main 

main: $(SRCS)
	$(CC) $(CFLAGSRELEASE)  $^ -o $@ -lm 

debug: $(SRCS)
	$(CC) $(CFLAGSDEBUG)  $^ -o $@ -lm 


