spm_encode \
    --model='/home/project/wasm-type-prediction/models/subword/type/names-filtered/500.model' \
    < "/home/project/wasm-type-prediction/data/dataset/names-filtered/split-by-dir-shuffle/return/test/wasm.txt" \
    > 'test.wasm.spm.txt'
# Then translate
CUDA_VISIBLE_DEVICES=1 \
onmt_translate \
    --src='test.wasm.spm.txt' \
    --model='/home/project/wasm-type-prediction/models/seq2seq/names-filtered/return/model_best.pt' \
    --output="predictions.model_best.spm.txt" \
    --log_file='predict.log' \
    --n_best=5 \
    --beam_size=5 \
    --report_time \
    --gpu=0 \
    --batch_size=100
# Use SentencePiece to decode the output back into regular tokens again, but only for models where the types are in a subword model.
if test -n "$type_spm"
then
    ~/wasm-type-prediction/sentencepiece/build/src/spm_decode \
        --model="/home/project/wasm-type-prediction/models/subword/type/names-filtered/500.model" \
        < 'predictions.model_best.spm.txt' \
        > 'predictions.model_best.txt'
fi
# Evaluate against ground-truth
/home/project/wasm-type-prediction/implementation/3-scripts/evaluate-predictions.py \
    --log="eval.log" \
    --predictions="predictions.model_best.txt" \
    --ground-truth="/home/project/wasm-type-prediction/data/dataset/names-filtered/split-by-dir-shuffle/return/test/type.txt"
