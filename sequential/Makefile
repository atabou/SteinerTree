
CC=gcc
COMMON=-Wall
SRC=./src
INC=./include
OBJ=./obj
LNK=-lm
BIN=./bin
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
NOCOLOR=\033[0m

COMPILE	= 	${OBJ}/main.o \
		  	${OBJ}/steiner.o \
		  	${OBJ}/graph.o \
		  	${OBJ}/llist.o \
			${OBJ}/heap.o \
			${OBJ}/fibheap.o \
			${OBJ}/set.o \
			${OBJ}/pair.o \
			${OBJ}/common.o \
			${OBJ}/bfs.o \
			${OBJ}/shortestpath.o

all: | pre-build ${BIN}/main
	@echo "${GREEN}Compilation done.${NOCOLOR}"

pre-build:
	@echo "Creating bin and obj folders if they don't exist."
	@mkdir -p "${OBJ}" "${BIN}"

${BIN}/main: ${COMPILE}
	${CC} ${COMMON} -I${INC} -o ${BIN}/main ${COMPILE} ${LNK}
	@echo "${YELLOW}Ignore Warnings${NOCOLOR}"

${OBJ}/main.o: ${SRC}/main.c ${INC}/graph.h ${INC}/set.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

${OBJ}/steiner.o: ${SRC}/steiner.c ${INC}/steiner.h ${INC}/bfs.h ${INC}/shortestpath.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

${OBJ}/graph.o: ${SRC}/graph.c ${INC}/graph.h ${INC}/common.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

${OBJ}/llist.o: ${SRC}/llist.c ${INC}/llist.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

${OBJ}/heap.o: ${SRC}/heap.c ${INC}/heap.h ${INC}/fibheap.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

${OBJ}/fibheap.o: ${SRC}/fibheap.c ${INC}/fibheap.h ${INC}/heap.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

${OBJ}/set.o: ${SRC}/set.c ${INC}/set.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

${OBJ}/pair.o: ${SRC}/pair.c ${INC}/pair.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

${OBJ}/common.o: ${SRC}/common.c ${INC}/common.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

${OBJ}/bfs.o: ${SRC}/bfs.c ${INC}/bfs.h ${INC}/graph.h ${INC}/set.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

${OBJ}/shortestpath.o: ${SRC}/shortestpath.c ${INC}/shortestpath.h ${INC}/graph.h ${INC}/heap.h ${INC}/pair.h
	${CC} ${COMMON} -Iinclude -c $< -o $@

clean:
	rm -rf ${OBJ} ${BIN}