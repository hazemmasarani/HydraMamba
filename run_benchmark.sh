conda activate mamba-ssm
cd /home/adutt/masarani/HydraMamba


DEV0=cuda:2
DEV1=cuda:3
PORT=12233
ITER=100

declare -A seqs
seqs[1]="512 256 128 64 32"
seqs[2]="256 128 64 32 16"
seqs[4]="128 64 32 16 8"
seqs[8]="64 32 16 8"
seqs[16]="32 16 8"
seqs[32]="16 8 4"
seqs[64]="8 4"

for BATCH in 1 2 4 8 16 32 64
do
    for SEQ in ${seqs[$BATCH]}
    do
        echo "Running batch=$BATCH seq_len=$SEQ"
        python benchmark.py -dev0 $DEV0 -dev1 $DEV1 -batch_size $BATCH -seq_len $SEQ -port_num $PORT -n_iter $ITER
    done
done