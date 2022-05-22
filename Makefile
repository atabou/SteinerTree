
CC=nvcc
COMMON= -g -G --Werror all-warnings -rdc=true -lcudadevrt -O3 -I./include -I${CONDA_PREFIX}/include
SRC=./src
INC=./include
OBJ=./obj
LNK=-lm -Xlinker=-rpath=${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib -lcugraph_c -lcugraph -lcugraph-ops++
BIN=./bin
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
NOCOLOR=\033[0m

COMPILE=steiner.o \
        graph.o \
	 	set.o \
        shortestpath.o \
		util.o \
		table.o \
		combination.cu.o \
		graph.cu.o \
		set.cu.o \
		table.cu.o \
		steiner.cu.o \

all: | pre-build ${BIN}/main
	@echo "${GREEN}Compilation done.${NOCOLOR}"

test: | pre-build ${BIN}/test
	@echo "${GREEN}Testing done.${NOCOLOR}"

pre-build:
	@echo "Creating bin and obj folders if they don't exist."
	@mkdir -p "${OBJ}" "${BIN}"

${BIN}/main: ./main.c $(patsubst %,${OBJ}/%,${COMPILE})
	${CC} ./main.c ${COMMON} -I/opt/cuda/include -o ${BIN}/main $(patsubst %,${OBJ}/%,${COMPILE}) ${LNK}
	@echo "${YELLOW}Ignore Warnings${NOCOLOR}"

${BIN}/test: ./test.c $(patsubst %,${OBJ}/%,${COMPILE})
	${CC} ./test.c ${COMMON} -I/opt/cuda/include -o ${BIN}/test $(patsubst %,${OBJ}/%,${COMPILE}) ${LNK}
	@echo "${YELLOW}Ignore Warnings${NOCOLOR}"

${OBJ}/%.cu.o: ${SRC}/%.cu ${INC}/%.cuda.h
	${CC} ${COMMON} -c $< -o $@

${OBJ}/%.o: ${SRC}/%.c ${INC}/%.h
	${CC} ${COMMON} -c $< -o $@

clean:
	rm -rf ${OBJ} ${BIN}
