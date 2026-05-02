import torch
import torch.nn as nn
from src.models.sasrec import SASRec
from torch.utils.data import DataLoader
from src.data.seqdataset import SequenceDataset
from src.data.seqdataset import collate_fn
import polars as pl
from src.utils.metrics import Metrics_k
from tqdm import tqdm

MAX_LEN=50

def sample_negatives(targets, num_items, num_negatives=1):
    
    B = targets.size(0)
    negatives = torch.randint(1, num_items + 1, (B, num_negatives))
    
    # Убеждаемся, что отрицательные не совпадают с положительными
    for i in range(B):
        for j in range(num_negatives):
            while negatives[i, j] == targets[i]:
                negatives[i, j] = torch.randint(1, num_items + 1, (1,))
    
    return negatives

def train_epoch_bce(model, dataloader, optimizer, num_items, device, num_negatives=3):
    """Обучение с Binary Cross Entropy и negative sampling."""
    model.train()
    total_loss = 0
    num_batches = 0
    bce_loss = nn.BCEWithLogitsLoss()

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        item_seq = batch['item_seq'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['target'].to(device)
        lengths = batch['lengths'].to(device)

        B = item_seq.size(0)
        
        # Получаем скоры для следующего товара
        scores = model.predict(item_seq, mask)  # (B, item_num+1)
        
        # Логиты для положительных товаров
        pos_logits = scores[torch.arange(B), targets]  # (B,)
        
        # Отрицательные примеры
        neg_targets = sample_negatives(targets, num_items, num_negatives).to(device)
        neg_logits = scores[torch.arange(B).unsqueeze(1), neg_targets]  # (B, num_negatives)
        
        # BCE Loss
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        all_labels = torch.cat([pos_labels.unsqueeze(1), neg_labels], dim=1)
        
        loss = bce_loss(all_logits, all_labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches

@torch.no_grad()
def evaluate(model, dataloader, num_items, device, ks=[5, 10, 20]):
    """Оценка модели с метриками HitRate@K, MRR@K, NDCG@K."""
    model.eval()
    
    # Создаём метрики для каждого K
    metrics = {k: Metrics_k(k) for k in ks}
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        item_seq = batch['item_seq'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['target'].to(device)
        
        # Получаем скоры для всех товаров
        scores = model.predict(item_seq, mask)  # (B, num_items+1)
        
        # Обновляем метрики для каждого K
        for k in ks:
            metrics[k].update(targets, scores)
    
    # Вычисляем финальные значения
    results = {}
    for k in ks:
        results[k] = metrics[k].compute()
    
    return results


class Args:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_units = 64
    maxlen = MAX_LEN
    dropout_rate = 0.2
    num_blocks = 2
    num_heads = 2
    norm_first = True
    batch_size = 128
    lr = 0.001
    num_epochs = 50
    num_negatives = 1  # количество отрицательных примеров
    temperature = 1.0  # для softmax (можно использовать для hard negative mining)
    ks = [5, 10, 20]
    patience = 10

def main():
    args = Args()
    device = args.device


    df = pl.read_parquet("dataset/processed/sequences.parquet")

    all_items = set()
    for row in df.to_dicts():
        all_items.update(row['item_sequence'])
    num_items = max(all_items)

    train_dataset = SequenceDataset(df, max_len=MAX_LEN, mode='train')
    val_dataset = SequenceDataset(df, max_len=MAX_LEN, mode='val')
    test_dataset = SequenceDataset(df, max_len=MAX_LEN, mode='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if args.device.type == 'cuda' else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model=SASRec(num_items,args).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)

    best_val_metric = 0
    best_epoch = 0
    patience_counter = 0

    # История обучения
    history = {
        'train_loss': [],
        'val_metrics': {k: {'hitrate': [], 'mrr': [], 'ndcg': []} for k in args.ks}
    }

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5)

    for epoch in range(args.num_epochs):
        # Обучение
        train_loss = train_epoch_bce(
            model, train_loader, optimizer, num_items,
            device, args.num_negatives
        )
        history['train_loss'].append(train_loss)

        # Валидация
        val_results = evaluate(model, val_loader, num_items, device, args.ks)

        # Сохраняем историю и выводим результаты
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")

        for k in args.ks:
            res = val_results[k]
            for metric_name in ['hitrate', 'mrr', 'ndcg']:
                history['val_metrics'][k][metric_name].append(res[metric_name])

            print(f"  K={k:2d} | HitRate: {res['hitrate']:.4f} | "
                  f"MRR: {res['mrr']:.4f} | NDCG: {res['ndcg']:.4f}")

        # Отслеживаем лучшую модель (по NDCG@10)
        current_metric = val_results[10]['ndcg']
        scheduler.step(current_metric)

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ndcg': current_metric,
                'args': args,
            }, 'sasrec_best_model.pt')
            print(f"  ✓ Saved best model (NDCG@10: {current_metric:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("\n" + "="*60)
    print("Loading best model for testing...")
    print("="*60)
    
    checkpoint = torch.load('sasrec_best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, num_items, device, args.ks)
    
    print("\n📊 Test Results:")
    print("─" * 40)
    for k in args.ks:
        res = test_results[k]
        print(f"K={k:2d}:")
        print(f"  HitRate@{k}: {res['hitrate']:.4f}")
        print(f"  MRR@{k}:     {res['mrr']:.4f}")
        print(f"  NDCG@{k}:    {res['ndcg']:.4f}")
        print()
if __name__=="__main__":
    main()