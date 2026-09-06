
import csv, io, math

csv_data = """batch_size,prompt_len,gen_len,num_requests,wall_clock_s,reported_tok_s,ttft_ms_p50,itl_ms_p50,e2e_ms_p95,preempted_seqs,kv_cache_util
1,512,256,1,10.94,70.2,71.1,43.48,12863.7,0,0.01
2,512,256,2,11.61,132.3,68.3,43.1,13045.7,0,0.01
4,512,256,4,11.77,261.0,71.5,45.34,13409.5,0,0.03
8,512,256,8,12.4,495.4,72.7,46.83,14137.1,0,0.06
16,512,256,16,13.91,883.2,71.7,48.33,15592.3,0,0.12
32,512,256,32,16.5,1489.6,68.3,56.17,18502.8,0,0.23
64,512,256,64,21.68,2267.3,73.3,67.91,24323.7,0,0.47
4,3584,512,4,28.98,565.4,483.2,51.33,32673.3,0,0.16
8,3584,512,8,36.3,902.6,519.0,62.26,39982.9,0,0.31
16,3584,512,16,49.97,1311.4,498.3,77.2,54602.1,0,0.62
24,3584,512,24,61.16,1607.4,500.5,96.07,69221.3,0,0.93
32,3584,512,32,94.71,1384.0,636.9,101.79,97465.7,7,0.97
48,3584,512,48,151.41,1298.5,955.4,100.0,105427.5,23,0.97
"""

rows = list(csv.DictReader(io.StringIO(csv_data.strip())))

def fv(r, col):
    return float(r[col])

def iv(r, col):
    return int(r[col])

long  = [r for r in rows if iv(r, 'prompt_len') == 3584]
short = [r for r in rows if iv(r, 'prompt_len') == 512]

# ============================================================
# B1a
# ============================================================
print("=" * 60)
print("B1a: KV-cache bytes per token")
print("=" * 60)
layers    = 28
kv_heads  = 8
head_dim  = 128
prec      = 2

bytes_per_tok = layers * 2 * kv_heads * head_dim * prec
print(f"formula : layers * 2 * kv_heads * head_dim * bytes_per_elem")
print(f"values  : {layers} * 2 * {kv_heads} * {head_dim} * {prec}")
print(f"result  : {bytes_per_tok} bytes/token  ({bytes_per_tok/1024:.2f} KB/token)")

# ============================================================
# B1b
# ============================================================
print()
print("=" * 60)
print("B1b: Max concurrent sequences at max_model_len=4096")
print("=" * 60)
gpu_mem_total   = 24.0e9
gpu_mem_util    = 0.92
usable_mem      = gpu_mem_total * gpu_mem_util
params          = 4.2e9
weight_bytes    = params * prec
non_kv_overhead = 1.6e9
kv_budget       = usable_mem - weight_bytes - non_kv_overhead
max_model_len   = 4096
kv_per_seq      = bytes_per_tok * max_model_len
max_seqs_f      = kv_budget / kv_per_seq
max_seqs        = math.floor(max_seqs_f)

print(f"usable_mem   = {gpu_mem_total/1e9} GB * {gpu_mem_util} = {usable_mem/1e9:.4f} GB")
print(f"weight_mem   = {params/1e9} B params * {prec} B/param = {weight_bytes/1e9:.4f} GB")
print(f"non_kv_oh    = {non_kv_overhead/1e9:.1f} GB")
print(f"kv_budget    = {usable_mem/1e9:.4f} - {weight_bytes/1e9:.4f} - {non_kv_overhead/1e9:.1f}")
print(f"             = {kv_budget/1e9:.6f} GB  ({kv_budget/1e6:.4f} MB)")
print(f"kv_per_seq   = {bytes_per_tok} bytes/tok * {max_model_len} tok")
print(f"             = {kv_per_seq/1e6:.4f} MB")
print(f"max_seqs     = {kv_budget/1e6:.4f} MB / {kv_per_seq/1e6:.4f} MB = {max_seqs_f:.6f}")
print(f"             => floor => {max_seqs} concurrent sequences")

# ============================================================
# B1 cross-check vs log
# ============================================================
print()
print("=" * 60)
print("B1 log cross-check")
print("=" * 60)
for r in long:
    bs   = iv(r, 'batch_size')
    kv   = fv(r, 'kv_cache_util')
    pre  = iv(r, 'preempted_seqs')
    print(f"  bs={bs:>2}: kv_util={kv:.2f}  preempted={pre:>2}")

# ============================================================
# B2
# ============================================================
print()
print("=" * 60)
print("B2: Scaling ratios")
print("=" * 60)
prev = None
for r in long:
    bs   = iv(r, 'batch_size')
    rtok = fv(r, 'reported_tok_s')
    if prev:
        ratio    = rtok / prev[1]
        bs_ratio = bs / prev[0]
        print(f"  bs {prev[0]:>2}->{bs:>2}: tok/s {prev[1]:>7.1f} -> {rtok:>7.1f}  "
              f"actual_ratio={ratio:.3f}  naive_expected={bs_ratio:.3f}")
    prev = (bs, rtok)

print()
print("Linear extrapolation from bs=24 (last clean, 0 preemptions, highest tok/s):")
bs24 = long[3]
assert iv(bs24, 'batch_size') == 24
bs24_toks = fv(bs24, 'reported_tok_s')
bs24_wall = fv(bs24, 'wall_clock_s')
for r in long[4:]:
    bs = iv(r, 'batch_size')
    exp_toks  = bs24_toks * (bs / 24)
    act_toks  = fv(r, 'reported_tok_s')
    tok_delta = (act_toks - exp_toks) / exp_toks * 100
    exp_wall  = bs24_wall * (bs / 24)
    act_wall  = fv(r, 'wall_clock_s')
    wall_oh   = (act_wall - exp_wall) / exp_wall * 100
    print(f"  bs={bs}: expected_tok_s={exp_toks:.2f}, actual={act_toks:.1f} => delta={tok_delta:+.1f}%")
    print(f"          expected_wall={exp_wall:.2f}s, actual={act_wall:.2f}s => wall_overhead={wall_oh:+.1f}%")

# ============================================================
# B3 formula reverse-engineering
# ============================================================
print()
print("=" * 60)
print("B3: Reverse-engineer reported_tok_s")
print("=" * 60)
print(f"  {'bs':>3} {'pl':>5} {'gl':>4} {'nr':>3} {'wall_s':>7} {'reported':>9}  "
      f"{'formula_A':>10}  {'formula_B':>10}  {'A_err%':>7}  {'B_err%':>7}")
for r in [rows[0], rows[4], rows[6], long[1], long[2], long[3]]:
    pl   = iv(r, 'prompt_len')
    gl   = iv(r, 'gen_len')
    nr   = iv(r, 'num_requests')
    ws   = fv(r, 'wall_clock_s')
    rep  = fv(r, 'reported_tok_s')
    fa   = gl * nr / ws
    fb   = (pl + gl) * nr / ws
    fa_e = (fa - rep) / rep * 100
    fb_e = (fb - rep) / rep * 100
    print(f"  {iv(r,'batch_size'):>3} {pl:>5} {gl:>4} {nr:>3} {ws:>7.2f} {rep:>9.1f}  "
          f"{fa:>10.2f}  {fb:>10.2f}  {fa_e:>+7.2f}%  {fb_e:>+7.2f}%")

print()
print("Explicit arithmetic for 3 rows:")
for r in [rows[0], rows[5], long[2]]:
    pl  = iv(r, 'prompt_len')
    gl  = iv(r, 'gen_len')
    nr  = iv(r, 'num_requests')
    ws  = fv(r, 'wall_clock_s')
    rep = fv(r, 'reported_tok_s')
    fa  = gl * nr / ws
    fb  = (pl + gl) * nr / ws
    print(f"  Row bs={iv(r,'batch_size')}, pl={pl}, gl={gl}, nr={nr}, wall={ws}:")
    print(f"    formula_A = {gl}*{nr}/{ws} = {fa:.4f}  err={((fa-rep)/rep*100):+.3f}%")
    print(f"    formula_B = ({pl}+{gl})*{nr}/{ws} = {fb:.4f}  err={((fb-rep)/rep*100):+.3f}%")

# ============================================================
# B3 honest goodput — bs=24, pl=3584
# ============================================================
print()
print("=" * 60)
print("B3: Honest goodput — bs=24, pl=3584")
print("=" * 60)
r24  = long[3]
gl   = iv(r24, 'gen_len')
nr   = iv(r24, 'num_requests')
ws   = fv(r24, 'wall_clock_s')
ttft = fv(r24, 'ttft_ms_p50') / 1000.0
itl  = fv(r24, 'itl_ms_p50') / 1000.0

m1 = gl * nr / ws
decode_wall = ws - ttft
m2 = gl * nr / decode_wall
m2b = nr / itl

print(f"  gen_len={gl}, num_requests={nr}, wall_clock_s={ws}")
print(f"  ttft_s={ttft:.5f}, itl_s={itl:.5f}")
print()
print(f"  Method 1: gen_len * num_requests / wall_clock_s")
print(f"    = {gl} * {nr} / {ws}")
print(f"    = {gl*nr} / {ws}")
print(f"    = {m1:.4f} tok/s")
print()
print(f"  Method 2: gen_len * num_requests / (wall_clock_s - ttft_s)")
print(f"    decode_wall = {ws} - {ttft:.5f} = {decode_wall:.5f} s")
print(f"    = {gl} * {nr} / {decode_wall:.5f}")
print(f"    = {m2:.4f} tok/s")
print()
print(f"  Method 2b: num_requests / itl_s  (concurrency / step_time)")
print(f"    = {nr} / {itl:.5f}")
print(f"    = {m2b:.4f} tok/s")
print()
pct_12  = abs(m1 - m2) / m1 * 100
pct_22b = abs(m2 - m2b) / m2 * 100
print(f"  M1 vs M2 difference:   {pct_12:.2f}%")
print(f"  M2 vs M2b difference:  {pct_22b:.2f}%")

print()
r48 = long[5]
assert iv(r48, 'batch_size') == 48
honest_48   = iv(r48, 'gen_len') * iv(r48, 'num_requests') / fv(r48, 'wall_clock_s')
reported_48 = fv(r48, 'reported_tok_s')
print(f"  REPORT prediction: batch48 => ~3200 tok/s")
print(f"  Actual reported_tok_s at bs=48: {reported_48} (off by {abs(reported_48-3200)/3200*100:.1f}%)")
print(f"  Actual honest goodput at bs=48: {honest_48:.2f} (off by {abs(honest_48-3200)/3200*100:.1f}%)")

reported_24 = fv(r24, 'reported_tok_s')
print()
print(f"  reported_tok_s at bs=24 = {reported_24}")
print(f"  honest goodput at bs=24  = {m1:.2f}")
print(f"  inflation factor: {reported_24/m1:.2f}x  "
      f"(because (prompt_len+gen_len)/gen_len = ({iv(r24,'prompt_len')}+{gl})/{gl} = {(iv(r24,'prompt_len')+gl)/gl:.4f}x)")
