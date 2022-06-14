
CC=nvcc
COMMON= -g -G -rdc=true -O3 -I./include --compiler-options -fPIC
SRC=./src
INC=./include
OBJ=./obj
LIB=./lib
LNK=-lm -Xlinker=-rpath=${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib -lcugraph_c -lcugraph -lcugraph-ops++
BIN=./bin
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
NOCOLOR=\033[0m

COMPILE=combination.o \
		graph.o \
	 	query.o \
		table.o \
        shortestpath.o \
		tree.o \
		steiner.cpu.o \
		steiner.gpu.o \
		util.o \

all: | pre-build ${BIN}/main
	@echo "${GREEN}Compilation done.${NOCOLOR}"

lib: | pre-build $(patsubst %,${OBJ}/%,${COMPILE})
	@${CC} --shared ${COMMON} -I/opt/cuda/include -o ${LIB}/libsteiner.so $(patsubst %,${OBJ}/%,${COMPILE}) ${LNK}
	@echo "${GREEN}Library creation done.${NOCOLOR}"

test: | pre-build ${BIN}/test 
	@echo "${GREEN}Test build done.${NOCOLOR}"

pre-build:
	@echo "Creating bin and obj folders if they don't exist."
	@mkdir -p "${OBJ}" "${BIN}" "${LIB}"

${BIN}/main: ./main.cu $(patsubst %,${OBJ}/%,${COMPILE})
	${CC} ./main.cu ${COMMON} -o ${BIN}/main $(patsubst %,${OBJ}/%,${COMPILE}) ${LNK}

${BIN}/test: ./test.cu ${LIB}/libsteiner.so
	${CC} ./test.cu ${COMMON} -o ${BIN}/test -L./lib -lsteiner

${LIB}/libsteiner.so: $(patsubst %,${OBJ}/%,${COMPILE})
	${CC} --shared ${COMMON} -o ${LIB}/libsteiner.so $(patsubst %,${OBJ}/%,${COMPILE}) ${LNK}

${OBJ}/combination.o: ${SRC}/combination.cu ${INC}/combination.hpp
	${CC} ${COMMON} -c $< -o $@

${OBJ}/graph.o: ${SRC}/graph.cu ${INC}/graph.hpp
	${CC} ${COMMON} -c $< -o $@

${OBJ}/query.o: ${SRC}/query.cu ${INC}/query.hpp ${INC}/graph.hpp ${INC}/table.hpp
	${CC} ${COMMON} -c $< -o $@

${OBJ}/table.o: ${SRC}/table.cu ${INC}/table.hpp
	${CC} ${COMMON} -c $< -o $@

${OBJ}/shortestpath.o: ${SRC}/shortestpath.cu ${INC}/shortestpath.hpp
	${CC} ${COMMON} -I${CONDA_PREFIX}/include -c $< -o $@

${OBJ}/tree.o: ${SRC}/tree.cu ${INC}/tree.hpp
	${CC} ${COMMON} -c $< -o $@

${OBJ}/steiner.cpu.o: ${SRC}/steiner.cpu.cu ${INC}/steiner.hpp ${INC}/combination.hpp ${INC}/util.hpp ${INC}/graph.hpp ${INC}/table.hpp ${INC}/query.hpp ${INC}/tree.hpp
	${CC} ${COMMON} -c $< -o $@

${OBJ}/steiner.gpu.o: ${SRC}/steiner.gpu.cu ${INC}/steiner.hpp ${INC}/combination.hpp ${INC}/util.hpp ${INC}/graph.hpp ${INC}/table.hpp ${INC}/query.hpp ${INC}/tree.hpp
	${CC} ${COMMON} -c $< -o $@

${OBJ}/util.o: ${SRC}/util.cu ${INC}/util.hpp
	${CC} ${COMMON} -c $< -o $@

clean:
	rm -rf ${OBJ} ${BIN} ${LIB}
