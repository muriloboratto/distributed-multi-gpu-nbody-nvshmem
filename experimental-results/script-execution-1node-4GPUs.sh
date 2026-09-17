#!/bin/sh

for i in 32768 65536 131072 262144
do
    for lib in MMMMMMMM CCCCCCCC NNNNNNNN SSSSSSSS
    do
        echo "Size = $i | Library = $lib"

        mpirun -np 1 ./n_body 0 $i $lib : \
               -np 1 ./n_body 1 $i $lib : \
               -np 1 ./n_body 2 $i $lib : \
               -np 1 ./n_body 3 $i $lib \
               >> result--32768-262144-1node-4GPUs-${lib}.txt
    done
done