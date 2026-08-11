#!/usr/bin/env python3
# trio_gvcf_cnv_mosaic.py  -- 可复用 skill 脚本
# 从 trio(三人各自单样本) GATK gVCF 的 FORMAT/DP 做：
#   A1) 读深度 CNV 全基因组扫描 (de novo CNV 优先，亦报绝对 CNV)
#   A8) 在指定 de novo/候选位点抓取父母+患儿 AD/DP，重查父母嵌合 / child 低AB
# 纯 Python stdlib(gzip)，不依赖 pysam。每个 gVCF 是单样本文件。
#
# 用法:
#   python trio_gvcf_cnv_mosaic.py --gvcf-dir DIR --out OUTDIR [--bin 5000]
#        [--targets targets.json]   # {"chr3:101565164":"TRMT10C", ...}
#        [--sex chroms auto]        # chrX/chrY 自动用 Father 比对
# 其中 DIR 含 Father.g.vcf.gz / Mother.g.vcf.gz / XY.g.vcf.gz (单样本)
import gzip, json, math, os, sys, time, argparse, statistics

# hg38 染色体长度
CHROM_LEN = {
 "chr1":248956422,"chr2":242193529,"chr3":198295559,"chr4":190214555,
 "chr5":181538259,"chr6":170805979,"chr7":159345973,"chr8":145138636,
 "chr9":138394717,"chr10":133797422,"chr11":135086622,"chr12":133275309,
 "chr13":114364328,"chr14":107043718,"chr15":101991189,"chr16":90338345,
 "chr17":83257441,"chr18":80373285,"chr19":58617616,"chr20":64444167,
 "chr21":46709983,"chr22":50818468,"chrX":156040895,"chrY":57227415,
 "chrM":16569,
}
AUTOSOMES = [f"chr{i}" for i in range(1,23)]
OFFSET = {}
cum = 0
for c in list(CHROM_LEN.keys()):
    OFFSET[c] = cum
    cum += (CHROM_LEN[c] // 5000) + 1
TOTAL_BINS0 = cum

def build_offsets(bin_size):
    off = {}; c = 0
    for ch in list(CHROM_LEN.keys()):
        off[ch] = c
        c += (CHROM_LEN[ch] // bin_size) + 1
    return off, c

def parse_dp(format_str, sample_str):
    cols = format_str.split(':'); vals = sample_str.split(':')
    try: return int(vals[cols.index('DP')])
    except (ValueError, IndexError): return 0

def ad_pair(format_str, sample_str):
    cols = format_str.split(':'); vals = sample_str.split(':')
    try:
        i = cols.index('AD'); parts = vals[i].split(',')
        if len(parts) >= 2: return int(parts[0]), int(parts[1])
    except (ValueError, IndexError): pass
    return None, None

def gt_of(format_str, sample_str):
    cols = format_str.split(':'); vals = sample_str.split(':')
    try: return vals[cols.index('GT')]
    except (ValueError, IndexError): return "."

def stream_sample(path, sample_name, bin_size, off, total_bins, sum_dp, sum_len, captured, targets):
    n = 0; t0 = time.time()
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line[0] == '#': continue
            n += 1
            if n % 5000000 == 0:
                el = time.time()-t0
                sys.stderr.write(f"  [{sample_name}] {n:,} lines, {el:.0f}s, {n/el/1000:.0f}k/s\n"); sys.stderr.flush()
            tabs = line.split('\t')
            if len(tabs) < 9: continue
            chrom = tabs[0]; pos = int(tabs[1]); info = tabs[7]; fmt = tabs[8]; samp = tabs[9]
            end = pos
            ei = info.find('END=')
            if ei != -1:
                ej = info.find(';', ei); ej = len(info) if ej == -1 else ej
                try: end = int(info[ei+4:ej])
                except: end = pos
            dp = parse_dp(fmt, samp)
            length = end - pos + 1
            if length < 1: length = 1
            bi = off[chrom] + (pos-1)//bin_size
            if 0 <= bi < total_bins:
                sum_dp[bi] += dp*length; sum_len[bi] += length
            key = f"{chrom}:{pos}"
            if key in targets:
                captured.setdefault(key, {})[sample_name] = {
                    "gt": gt_of(fmt, samp), "ad": list(ad_pair(fmt, samp)), "dp": dp, "end": end}
    sys.stderr.write(f"  [{sample_name}] DONE {n:,} lines in {time.time()-t0:.0f}s\n")

def bin_to_chrompos(bi, bin_size, off):
    for c in CHROM_LEN:
        if off[c] <= bi < off[c] + (CHROM_LEN[c]//bin_size)+1:
            return c, (bi-off[c])*bin_size+1
    return "chr?", bi*bin_size+1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvcf-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", type=int, default=5000)
    ap.add_argument("--targets", default=None, help="json {chr:pos: genename}")
    ap.add_argument("--ndd-loci", default=None, help="可选 json [(name,chrom,start,end),...]")
    args = ap.parse_args()
    bin_size = args.bin
    off, total_bins = build_offsets(bin_size)
    files = {"Father": os.path.join(args.gvcf_dir,"Father.g.vcf.gz"),
             "Mother": os.path.join(args.gvcf_dir,"Mother.g.vcf.gz"),
             "XY":     os.path.join(args.gvcf_dir,"XY.g.vcf.gz")}
    targets = {}
    if args.targets:
        with open(args.targets) as fh: targets = json.load(fh)
    ndd_loci = []
    if args.ndd_loci:
        with open(args.ndd_loci) as fh: ndd_loci = json.load(fh)

    sum_dp = {s:[0.0]*total_bins for s in files}
    sum_len = {s:[0.0]*total_bins for s in files}
    captured = {}
    for s,p in files.items():
        stream_sample(p, s, bin_size, off, total_bins, sum_dp[s], sum_len[s], captured, targets)

    depth = {}
    for s in files:
        d = [0.0]*total_bins; sl = sum_len[s]
        for i in range(total_bins):
            if sl[i] > 0: depth[s][i] = sum_dp[s][i]/sl[i]
    medians = {}
    for s in files:
        vals = []
        for c in AUTOSOMES:
            o = off[c]; ln = (CHROM_LEN[c]//bin_size)+1
            for i in range(o,o+ln):
                if sum_len[s][i] > 0 and depth[s][i] > 0: vals.append(depth[s][i])
        medians[s] = statistics.median(vals) if vals else 1.0
        sys.stderr.write(f"  median depth {s} = {medians[s]:.1f}\n")
    nd = {}
    for s in files:
        m = medians[s]
        nd[s] = [ (depth[s][i]/m) if depth[s][i]>0 else 0.0 for i in range(total_bins) ]

    calls = []
    for c in (AUTOSOMES + ["chrX","chrY"]):
        o = off[c]; ln = (CHROM_LEN[c]//bin_size)+1
        run = None
        for i in range(o,o+ln):
            cf = nd["XY"][i]
            pf = nd["Father"][i] if c in ("chrX","chrY") else (nd["Father"][i]+nd["Mother"][i])/2.0
            slxy = sum_len["XY"][i]; slp = (sum_len["Father"][i]+sum_len["Mother"][i])/2.0
            M = math.log2(cf/pf) if (cf>0 and pf>0 and slxy>0.5*bin_size and slp>0.5*bin_size) else 0.0
            sign = 1 if M>0.4 else (-1 if M<-0.4 else 0)
            if sign != 0:
                if run and run[0]==sign: run[1].append((i,M,cf,pf))
                else:
                    if run: calls.append(run)
                    run = (sign,[(i,M,cf,pf)])
            else:
                if run: calls.append(run); run=None
        if run: calls.append(run)

    cnv = []
    for sign, segs in calls:
        if len(segs) < 3: continue
        chrom, spos = bin_to_chrompos(segs[0][0], bin_size, off)
        _, epos = bin_to_chrompos(segs[-1][0], bin_size, off); epos += bin_size-1
        meanM = sum(s[1] for s in segs)/len(segs)
        mean_cf = sum(s[2] for s in segs)/len(segs)
        mean_pf = sum(s[3] for s in segs)/len(segs)
        flags = [nm for (nm,fc,fs,fe) in ndd_loci if fc==chrom and not (epos<fs or spos>fe)]
        cnv.append({"chrom":chrom,"start":spos,"end":epos,
                    "type":"DUP" if sign>0 else "DEL","n_bins":len(segs),
                    "mean_log2ratio":round(meanM,3),"child_norm":round(mean_cf,3),
                    "parent_norm":round(mean_pf,3),"approx_copynumber":round(2*(2**meanM),2),
                    "ndd_overlap":flags})
    cnv.sort(key=lambda x: abs(x["mean_log2ratio"]), reverse=True)

    ndd_report = []
    for (name,fc,fs,fe) in ndd_loci:
        if fc not in off: continue
        ms = []; o=off[fc]; ln=(CHROM_LEN[fc]//bin_size)+1
        for i in range(o,o+ln):
            pos=(i-off[fc])*bin_size+1
            if pos<fs or pos>fe: continue
            cf=nd["XY"][i]; pf=nd["Father"][i] if fc in ("chrX","chrY") else (nd["Father"][i]+nd["Mother"][i])/2.0
            if cf>0 and pf>0: ms.append(math.log2(cf/pf))
        if ms: ndd_report.append({"locus":name,"chrom":fc,
                                 "m_log2_median":round(statistics.median(ms),3),
                                 "m_log2_min":round(min(ms),3),"m_log2_max":round(max(ms),3)})

    mosaic = []
    for key, gene in targets.items():
        rec = captured.get(key, {}); row={"locus":key,"gene":gene}
        for s in ["XY","Father","Mother"]:
            d = rec.get(s)
            if not d: row[s]=None; continue
            r,a = (d["ad"][0], d["ad"][1]) if d["ad"][0] is not None else (None,None)
            ab = round(a/(r+a),3) if (r is not None and a is not None and (r+a)>0) else None
            row[s]={"gt":d["gt"],"ref_ad":r,"alt_ad":a,"alt_frac":ab,"dp":d["dp"]}
        f=row.get("Father"); m=row.get("Mother"); xy=row.get("XY")
        palt = (f["alt_ad"] if f and f["alt_ad"] else 0)+(m["alt_ad"] if m and m["alt_ad"] else 0)
        if xy and xy["alt_frac"] is not None and xy["alt_frac"]<0.30:
            v="child低AB(<0.3)：非真胚系de novo典型(~0.5)；提示患儿体细胞嵌合或比对/calling伪影"
        elif xy and xy["alt_frac"] is not None:
            v="child AB正常(~0.5)"
        else: v="未抓到child记录"
        v += ("；父母alt=0→无可检测的父母生殖腺嵌合(需BAM/ddPCR定低比例嵌合)" if palt==0
              else f"；父母合计alt={palt}→疑父母低比例嵌合")
        row["verdict"]=v; mosaic.append(row)

    out = {"params":{"bin":bin_size,"medians":medians},
           "cnv_calls":cnv,"ndd_locus_depth":ndd_report,"mosaic_recheck":mosaic}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out,"gvcf_cnv_mosaic.json"),"w") as fh: json.dump(out,fh,indent=2)
    with open(os.path.join(args.out,"gvcf_cnv_calls.tsv"),"w") as fh:
        fh.write("chrom\tstart\tend\ttype\tn_bins\tmean_log2ratio\tapprox_CN\tndd_overlap\n")
        for c in cnv:
            fh.write(f"{c['chrom']}\t{c['start']}\t{c['end']}\t{c['type']}\t{c['n_bins']}\t{c['mean_log2ratio']}\t{c['approx_copynumber']}\t{';'.join(c['ndd_overlap'])}\n")
    sys.stderr.write(f"WROTE gvcf_cnv_mosaic.json + .tsv ({len(cnv)} calls)\n")

if __name__ == "__main__":
    main()
