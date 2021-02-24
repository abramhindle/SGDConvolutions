pushd sounds.out
for x in `seq -f '%05g' 1 500`; do echo $x; sox snippet-adamax.00*.$x.wav $x.wav ; done
