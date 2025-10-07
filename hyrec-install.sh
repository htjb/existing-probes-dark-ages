git clone https://github.com/nanoomlee/HYREC-2.git
cd HYREC-2
gcc -lm -O3 hyrectools.c helium.c hydrogen.c history.c energy_injection.c hyrec.c -o hyrec
cd ../