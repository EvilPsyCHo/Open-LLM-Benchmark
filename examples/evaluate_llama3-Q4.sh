
benchmark.retrieval  \
--save_dir result/llama3-Q4_K_M-llamacpp \
--test_model /data/hf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
--test_model_loader llama_cpp \
--eval_model gpt-4-turbo \
--eval_model_loader openai \
--ctx_len_max 8000 \
--ctx_bins 10 \
--depth_bins 10

benchmark.agent  \
--save_dir result/llama3-Q4_K_M-llamacpp \
--test_model /data/hf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
--test_model_loader llama_cpp \
--eval_model gpt-4-turbo \
--eval_model_loader openai

benchmark.format_output  \
--save_dir result/llama3-Q4_K_M-llamacpp \
--test_model /data/hf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
--test_model_loader llama_cpp \
--eval_model gpt-4-turbo \
--eval_model_loader openai
