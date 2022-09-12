import os 
import sys
import shutil
_, out_path, seq_path, trained_model = sys.argv

def bpseq2fasta(fname, out_path):
    seq = "".join([line.split()[1] for line in open(fname)])
    id = fname.split("/")[-1].split(".")[0]
    out_file = f"{out_path}{id}.txt"
    with open(out_file, "w") as fout:
        fout.write(f">{id}\n")
        fout.write(seq)
    
    return id, out_file

if not os.path.isdir(out_path):
    os.mkdir(out_path)

failed = []
for f in os.listdir(seq_path):
    id, out_file = bpseq2fasta(f"{seq_path}{f}", out_path)
    
    os.system(f"python ufold_predict.py --trained_model {trained_model} --pred_file {out_file}")
    try:
        shutil.copyfile(f"results/save_ct_file/{id}.ct", f"{out_path}/{id}.ct")
    except FileNotFoundError:
        failed.append(id)
print("failed sequences", len(failed))
print(failed)