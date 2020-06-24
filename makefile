CC = mpicc
LIB = -I${MKLROOT}/include  -I${MKLROOT}/include   ${MKLROOT}/lib/libmkl_intel_lp64.a     ${MKLROOT}/lib/libmkl_core.a ${MKLROOT}/lib/libmkl_sequential.a

all: dave_qn dave_rpg dane giant 

dave_qn: dave_qn.c 
	$(CC) -o dave_qn.o dave_qn.c  $(LIB)

dave_rpg: dave_rpg.c
	$(CC) -o dave_rpg.o  dave_rpg.c  $(LIB)

dane: dane.c
	$(CC) -o dane.o  dane.c  $(LIB)

giant: giant.c
	$(CC) -o giant.o  giant.c  $(LIB)

clean :
	rm *.o
