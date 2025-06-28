CC=gcc
CFLAGS=-Wall -O2 -lm

SRCS = main.c tensor.c conv2d.c activation.c dense.c flatten.c loss.c
OBJS = $(SRCS:.c=.o)
TARGET = cnn.out

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
