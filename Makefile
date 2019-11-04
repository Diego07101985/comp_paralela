

install:
	- virtualenv ../mestrado_cefet
	- ( \
       . bin/activate; \
       pip3 install -U -r requirements.txt; \
    )
run:
	- ( \
       . bin/activate; \
    mpirun --hostfile hostfile -n 4 python main.py \
    )