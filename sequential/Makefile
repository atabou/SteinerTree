
SRC=./src
INC=./include
OBJ=./obj
LNK=-lm
BIN=./bin

COMPILE=${OBJ}/main.o ${OBJ}/graph.o ${OBJ}/heap.o ${OBJ}/fibheap.o ${OBJ}/set.o ${OBJ}/llist.o


${BIN}/main: ${COMPILE}
	gcc -I${INC} -o ${BIN}/main ${COMPILE} ${LNK}

${OBJ}/main.o: ${SRC}/main.c ${INC}/graph.h ${INC}/set.h
	gcc -Iinclude -c $< -o $@

${OBJ}/graph.o: ${SRC}/graph.c ${INC}/graph.h ${INC}/heap.h ${INC}/llist.h
	gcc -Iinclude -c $< -o $@

${OBJ}/llist.o: ${SRC}/llist.c ${INC}/llist.h
	gcc -Iinclude -c $< -o $@

${OBJ}/heap.o: ${SRC}/heap.c ${INC}/heap.h ${INC}/fibheap.h
	gcc -Iinclude -c $< -o $@

${OBJ}/fibheap.o: ${SRC}/fibheap.c ${INC}/fibheap.h ${INC}/heap.h
	gcc -Iinclude -c $< -o $@

${OBJ}/set.o: ${SRC}/set.c ${INC}/set.h
	gcc -Iinclude -c $< -o $@


clean:
	rm ${OBJ}/* ${BIN}/*