import os 
import glob 
import shutil 
import random 
import multiprocessing as mp 



def per_barcode(barcode, src_dir, dst_dir):
    src_barcode_dir = os.path.join(src_dir, barcode)
    save_dir = os.path.join(dst_dir, barcode)
    os.makedirs(save_dir, exist_ok=True)
    trains = glob.glob(f"{src_barcode_dir}/train*")
    vals = glob.glob(f"{src_barcode_dir}/val*")
    train_100 = random.sample(trains, k=90)
    val_100 = random.sample(vals, k=10)
    for train in train_100:
        basename = os.path.basename(train)
        shutil.copy(train, os.path.join(save_dir, basename))

    for val in val_100:
        basename = os.path.basename(val)
        shutil.copy(val, os.path.join(save_dir, basename))


CPU_CORE = os.cpu_count()
pool = mp.Pool(CPU_CORE-1)

src_dir = "/home/walter/big_daddy/onboard_big_fourview"
dst_dir = "/home/walter/big_daddy/onboard_big_fourview_100_per_product"

barcodes = os.listdir(src_dir)
for barcode in barcodes:
    pool.apply_async(per_barcode, args=(barcode, src_dir, dst_dir))

pool.close()
pool.join()
    
    