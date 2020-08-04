HIERTYPE=..
INDATA=../corpus/OntoNotes
OUTDATA=./prepared_data/OntoNotes
MODELDIR=./hiertype_onto
CONTEXTUALIZER=elmo-original
TOKENIZATION_METHOD=subword

wget http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip
unzip corpus.zip

mkdir -p $OUTDATA
python scripts/reformat_shimaoka.py $INDATA $OUTDATA

for partition in train dev test; do
    python $HIERTYPE/hiertype/commands/prepare_data_hdf5.py \
        --input_fp $OUTDATA/${partition}.txt \
        --output $OUTDATA/${partition}.hdf5 \
        --model $CONTEXTUALIZER \
        --unit $TOKENIZATION_METHOD \
        --layers [0,1,2] \
        --batch_size 16
done

mkdir -p $MODELDIR
python $HIERTYPE/hiertype/commands/train.py \
    --train $OUTDATA/train.hdf5 \
    --dev $OUTDATA/dev.hdf5 \
    --ontology $OUTDATA/ontology.txt \
    --out $MODELDIR \
    --margins [3,2,1] \
    --strategies [other,none,none] \
    --delta [0,1,2.5] \
    --max_branching_factors [1,1,1]

