
CC=nvcc
COMMON= -g -G -rdc=true -lcudadevrt -O3 -I./include
SRC=./src
INC=./include
OBJ=./obj
LNK=-lm -Xlinker=-rpath=${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib -lcugraph_c -lcugraph -lcugraph-ops++
BIN=./bin
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
NOCOLOR=\033[0m

COMPILE=combination.o \
		graph.o \
	 	query.o \
        shortestpath.o \
		tree.o \
		steiner.cpu.o \
		steiner.gpu.o \
		util.o \

all: | pre-build ${BIN}/main
	@echo "${GREEN}Compilation done.${NOCOLOR}"

test: | pre-build ${BIN}/test
	@echo "${GREEN}Test build done.${NOCOLOR}"
	@./bin/test
	@echo "${GREEN}Testing done.${NOCOLOR}"

pre-build:
	@echo "Creating bin and obj folders if they don't exist."
	@mkdir -p "${OBJ}" "${BIN}"

${BIN}/main: ./main.cu $(patsubst %,${OBJ}/%,${COMPILE})
	${CC} ./main.cu ${COMMON} -I/opt/cuda/include -o ${BIN}/main $(patsubst %,${OBJ}/%,${COMPILE}) ${LNK}
	@echo "${YELLOW}Ignore Warnings${NOCOLOR}"

${BIN}/test: ./test.cu $(patsubst %,${OBJ}/%,${COMPILE})
	${CC} ./test.cu ${COMMON} -I/opt/cuda/include -o ${BIN}/test $(patsubst %,${OBJ}/%,${COMPILE}) ${LNK}
	@echo "${YELLOW}Ignore Warnings${NOCOLOR}"

${OBJ}/combination.o: ${SRC}/combination.cu ${INC}/combination.h
	${CC} ${COMMON} -c $< -o $@

${OBJ}/graph.o: ${SRC}/graph.cu ${INC}/graph.h
	${CC} ${COMMON} -c $< -o $@

${OBJ}/query.o: ${SRC}/query.cu ${INC}/query.h ${INC}/graph.h ${INC}/table.h
	${CC} ${COMMON} -c $< -o $@

${OBJ}/shortestpath.o: ${SRC}/shortestpath.cu ${INC}/shortestpath.h
	${CC} ${COMMON} -I${CONDA_PREFIX}/include -c $< -o $@

${OBJ}/tree.o: ${SRC}/tree.cu ${INC}/tree.hpp
	${CC} ${COMMON} -c $< -o $@

${OBJ}/steiner.cpu.o: ${SRC}/steiner.cpu.cu ${INC}/steiner.h ${INC}/combination.h ${INC}/util.h ${INC}/graph.h ${INC}/table.h ${INC}/query.h ${INC}/tree.hpp
	${CC} ${COMMON} -c $< -o $@

${OBJ}/steiner.gpu.o: ${SRC}/steiner.gpu.cu ${INC}/steiner.h ${INC}/combination.h ${INC}/util.h ${INC}/graph.h ${INC}/table.h ${INC}/query.h ${INC}/tree.hpp
	${CC} ${COMMON} -c $< -o $@

${OBJ}/table.o: ${SRC}/table.cu ${INC}/table.h
	${CC} ${COMMON} -c $< -o $@

${OBJ}/util.o: ${SRC}/util.cu ${INC}/util.h
	${CC} ${COMMON} -c $< -o $@

clean:
	rm -rf ${OBJ} ${BIN}
