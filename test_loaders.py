import time
from jepa_robotics.data.dataset_loaders import BridgeDataLoader, SO100DataLoader

def test_loader(name, loader):
    print(f"\nTesting {name}...")
    start_time = time.time()
    count = 0
    iterator = loader.load(split="train")
    
    # Warmup
    try:
        next(iterator)
    except Exception as e:
        print(e)
        return
        
    start_time = time.time()
    for batch in iterator:
        count += 1
        if count >= 20:
            break
            
    elapsed = time.time() - start_time
    print(f"{name} speed: {count / elapsed:.2f} batches/s")

b_loader = BridgeDataLoader(batch_size=128, seq_len=6, sample_fraction=0.3)
s_loader = SO100DataLoader(batch_size=128, seq_len=6, sample_fraction=0.3)

test_loader("BridgeDataLoader", b_loader)
test_loader("SO100DataLoader", s_loader)
