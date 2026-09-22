# Ensure the results directory exists
mkdir -p bench2_results

# Execute Estonia Loop (10 Runs)
for i in {1..10}; do
  echo "=== Running Estonia Test $i/10 ===" | tee -a bench2_results/estonia_summary.log
  python3 llm_decode_bench.py \
    --host http://localhost \
    --port 8012 \
    --api-key private_api_key \
    --model GLM-5.3-Flash-EXL3-3.5bpw-fp8 \
    --test-profile estonia \
    >> bench2_results/estonia_run_$i.log 2>&1
  
  # Scrape accuracy results into the summary file
  grep -E "PASS|DECOY|NOT_STATED|FAIL|ERROR|accuracy" bench2_results/estonia_run_$i.log >> bench2_results/estonia_summary.log
done

# Execute Hotel-Lights Loop (10 Runs)
for i in {1..10}; do
  echo "=== Running Hotel-Lights Test $i/10 ===" | tee -a bench2_results/hotel_summary.log
  python3 llm_decode_bench.py \
    --host http://localhost \
    --port 8012 \
    --api-key private_api_key \
    --model GLM-5.3-Flash-EXL3-3.5bpw-fp8 \
    --test-profile hotel-lights \
    >> bench2_results/hotel_run_$i.log 2>&1
  
  grep -E "PASS|FAIL|ERROR|accuracy|Final" bench2_results/hotel_run_$i.log >> bench2_results/hotel_summary.log
done
