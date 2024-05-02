SAVE_DIR=result/llama3-vllm
TEST_MODEL=/data/hf/Meta-Llama-3-8B-Instruct
TEST_LOADER=vllm
EVAL_MODEL=gpt-4-turbo
EVAL_LOADER=openai


benchmark.retrieval  \
--save_dir $SAVE_DIR \
--test_model $TEST_MODEL \
--test_model_loader $TEST_LOADER \
--eval_model $EVAL_MODEL \
--eval_model_loader $EVAL_LOADER \
--ctx_len_max 8000 \
--ctx_bins 10 \
--depth_bins 10

benchmark.agent  \
--save_dir $SAVE_DIR \
--test_model $TEST_MODEL \
--test_model_loader $TEST_LOADER \
--eval_model $EVAL_MODEL \
--eval_model_loader $EVAL_LOADER

benchmark.format_output  \
--save_dir $SAVE_DIR \
--test_model $TEST_MODEL \
--test_model_loader $TEST_LOADER \
--eval_model $EVAL_MODEL \
--eval_model_loader $EVAL_LOADER
