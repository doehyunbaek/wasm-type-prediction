# echo 'Encoding...'
# spm_encode \
#     --model='/home/project/wasm-type-prediction/snowwhite/subword/wasm/500.model' \
#     < "/home/project/wasm-type-prediction/data/test/wasm.txt" \
#     > 'test.wasm.spm.txt'

echo 'Translting...'
# CUDA_VISIBLE_DEVICES=1 \
python ~/OpenNMT-py/translate.py \
    --src='/home/project/wasm-type-prediction/nllb/test.wasm.spm.txt' \
    --model="/home/project/wasm-type-prediction/nllb/save/nllb-200-wasm_step_11000.pt" \
    --output="/home/project/wasm-type-prediction/nllb/predictions.model_best.spm.txt" \
    --log_file='infer.log' \
    --n_best=5 \
    --beam_size=5 \
    --report_time \
    --gpu=0 \
    --batch_size=2

echo 'Decoding...'
spm_decode \
    --model="/home/project/wasm-type-prediction/snowwhite/subword/type/500.model" \
    < '/home/project/wasm-type-prediction/nllb/predictions.model_best.spm.txt' \
    > '/home/project/wasm-type-prediction/nllb/predictions.model_best.txt'

echo 'Evaluating...'
# /home/project/wasm-type-prediction/evaluate-predictions.py \
#     --log="eval.log" \
#     --predictions="predictions.model_best.txt" \
#     --ground-truth="/home/project/wasm-type-prediction/data/test/type.txt"

# match format
/home/project/wasm-type-prediction/snowwhite/translate.py \
    --input="/home/project/wasm-type-prediction/nllb/predictions.model_best.txt" \
    --output="/home/project/wasm-type-prediction/nllb/predictions.model_best_translate.txt"

/home/project/wasm-type-prediction/snowwhite/evaluate-predictions.py \
    --log="eval.log" \
    --predictions="/home/project/wasm-type-prediction/nllb/predictions.model_best_translate.txt" \
    --ground-truth="/home/project/wasm-type-prediction/data/test/type.txt"
