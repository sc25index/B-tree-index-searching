
#!/bin/bash
inputSize=100000000
keySize=100000000
randomFilePath=/home/napath/RandomGenerator/
RandomFileName=Random
while [ $keySize -le 1000000000 ]
do
    i=1
    while [ $i -le 4 ]
    do
    	ii=$(($i + 1))
    	localinputSize=$(( $inputSize / 1000000 ))
    	randomFileNameInput="${randomFilePath}${RandomFileName}${localinputSize}M_${i}.bin"
    	localKeySize=$(( $keySize / 1000000 ))
    	randomFileNameKey="${randomFilePath}${RandomFileName}${localKeySize}M_${ii}.bin"
    	echo "InputSize ${localinputSize}M KeySize ${localKeySize}M Dataset $i"
    	echo $randomFileNameInput
    	echo $randomFileNameKey
    	echo ./btree $inputSize $randomFileNameInput $keySize $randomFileNameKey
    	./btree $inputSize $randomFileNameInput $keySize $randomFileNameKey
    	i=$(($i+1))
    	echo "========================================="
    done
    keySize=$(( $keySize + 100000000 ))
done