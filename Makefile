

install:
	- virtualenv ../mestrado-comp-paralela
	- ( \
       . bin/activate; \
       pip3 install -U -r requirements.txt; \
    )
run-mpi:
	- ( \
       . bin/activate; \
         mpirun -n 11 --hostfile hostfile python3 kmeans_mpi.py  \
    )

run:
	- ( \
       . bin/activate; \
        python3 main.py  \
    )