mkdir -p bench1_results

for i in {1..3}; do
  echo "=== Small Benchmark Batch Run $i/3 ===" | tee -a bench1_results/summary.log
  python3 llm_decode_bench.py \
    --host http://localhost \
    --port 8012 \
    --api-key private_api_key \
    --model GLM-5.3-Flash-EXL3-3.5bpw-fp8 \
    --test-profile estonia \
    --temperature 0.0 \
    >> bench1_results/run_$i.log 2>&1
  
  grep -E "PASS|DECOY|FAIL|ERROR|accuracy" bench1_results/run_$i.log >> bench1_results/summary.log
done
