mpimake: dave_qn.c 
	mpicc -o dave_qn.o dave_qn.c  -I${MKLROOT}/include  -I${MKLROOT}/include   ${MKLROOT}/lib/libmkl_intel_lp64.a     ${MKLROOT}/lib/libmkl_core.a ${MKLROOT}/lib/libmkl_sequential.a

dave_rpg.o: dave_rpg.c
	mpicc -o dave_rpg.o  dave_rpg.c  -I${MKLROOT}/include   -I${MKLROOT}/include       ${MKLROOT}/lib/libmkl_intel_lp64.a     ${MKLROOT}/lib/libmkl_core.a ${MKLROOT}/lib/libmkl_sequential.a

dane.o: dane.c
	mpicc -o dane.o  dane.c  -I${MKLROOT}/include   -I${MKLROOT}/include       ${MKLROOT}/lib/libmkl_intel_lp64.a     ${MKLROOT}/lib/libmkl_core.a ${MKLROOT}/lib/libmkl_sequential.a

simplesgd.o: simplesgd.c
	mpicc -o simplesgd.o  simplesgd.c  -I${MKLROOT}/include   -I${MKLROOT}/include       ${MKLROOT}/lib/libmkl_intel_lp64.a     ${MKLROOT}/lib/libmkl_core.a ${MKLROOT}/lib/libmkl_sequential.a

clean :
	rm dave_qn.o dave_rpg.o dane.o
