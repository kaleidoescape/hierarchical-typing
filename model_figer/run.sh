HIERTYPE=..
INDATA=./corpus/Wiki
OUTDATA=./prepared_data/Wiki
MODELDIR=./hiertype_figer

wget -nc http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip
unzip corpus.zip

mkdir -p $OUTDATA
python scripts/reformat_shimaoka.py $INDATA $OUTDATA

for partition in train dev test; do
    echo "Reformatting $partition ..."
    python $HIERTYPE/hiertype/commands/prepare_data_hdf5.py \
        --input_fp $OUTDATA/${partition}.txt \
        --output $OUTDATA/${partition}.hdf5 \
        --model elmo-original \
        --unit word \
        --layers [0,1,2] \
        --batch_size 256
done

mkdir -p $MODELDIR
python $HIERTYPE/hiertype/commands/train.py \
    --train $OUTDATA/train.hdf5 \
    --dev $OUTDATA/dev.hdf5 \
    --ontology $OUTDATA/ontology.txt \
    --out $MODELDIR \
    --margins [3,2,1] \
    --strategies [top,none] \
    --delta [3,1] \
    --max_branching_factors [2,1] \
    --dropout_rate 0.1 \
    --emb_dropout_rate 0.5 \
    --threshold_ratio 0.2 \
    --relation_constraint_coef 0.1 \
    --regularizer 0.001 \
    --with_other True \
    --cuda_device 0 1 2 3 4 5 6 7

mkdir -p $MODELDIR/evaluation
python $HIERTYPE/hiertype/commands/infer.py \
    --model $MODELDIR \
    --test $OUTDATA/test.hdf5 \
    --out $MODELDIR/evaluation/ \
    --margins [3,2,1] \
    --max_branching_factors [2,1] \
    --strategies [top,none] \
    --delta [3,1] \

