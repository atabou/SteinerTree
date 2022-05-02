
CC=nvcc
COMMON= -g -G --Werror all-warnings -rdc=true -lcudadevrt -O3
SRC=./src
INC=./include
OBJ=./obj
LNK=-lm
BIN=./bin
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
NOCOLOR=\033[0m

COMPILE=steiner.o \
        graph.o \
        llist.o \
        heap.o \
        fibheap.o \
	 	set.o \
        pair.o \
        shortestpath.o \
		util.o \
		table.o \
		combination.cu.o \
		graph.cu.o \
		llist.cu.o \
		set.cu.o \
		table.cu.o \
		steiner.cu.o \
		steiner1.cu.o \
		

all: | pre-build ${BIN}/main
	@echo "${GREEN}Compilation done.${NOCOLOR}"

pre-build:
	@echo "Creating bin and obj folders if they don't exist."
	@mkdir -p "${OBJ}" "${BIN}"

${BIN}/main: ./main.c $(patsubst %,${OBJ}/%,${COMPILE})
	${CC} ./main.c ${COMMON} -I${INC} -o ${BIN}/main $(patsubst %,${OBJ}/%,${COMPILE}) ${LNK}
	@echo "${YELLOW}Ignore Warnings${NOCOLOR}"

${OBJ}/%.cu.o: ${SRC}/%.cu ${INC}/%.cuda.h
	${CC} ${COMMON} -I${INC} -c $< -o $@

${OBJ}/%.o: ${SRC}/%.c ${INC}/%.h
	${CC} ${COMMON} -I${INC} -c $< -o $@

clean:
	rm -rf ${OBJ} ${BIN}
