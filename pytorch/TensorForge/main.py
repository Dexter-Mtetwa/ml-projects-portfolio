import torch
import torch.nn as nn
from data.generators import generate_spiral, generate_xor
from data.datasets import create_dataloader
from models.mlp import MLP, MLPSequential
from models.cnn import SimpleCNN
from training.trainer import train_epoch, eval_model
from training.optimizers import SGDFullFromScratch
from utils.console import print_loss, print_accuracy
from utils.persistence import save_model, load_model
import os

device = 'cude' if torch.cuda.is_available() else 'cpu'
print(f'using {device}')

os.makedirs('saved', exist_ok=True)

def menu():
    model = None
    while True:
        print('\n=== TensorForge CLI ===')
        print('1. Generate Spiral Dataset')
        print('2. Generate XOR Dataset')
        print('3. Train MLP from scratch')
        print('4. Train MLP with nn.Sequential')
        print('5. Train CNN on fake images')
        print('6. Save Model')
        print('7. Load Model')
        print('8. Test LinearFromScratch vs nn.Linear')
        print('9. Exit')

        choice = input("Choice: ").strip()

        # 1. Generate Spiral Dataset
        if choice == '1':
            X, y = generate_spiral(n_points=3000)
            torch.save({'X':X, 'y':y}, 'saved//spiral.pt')
            print(f'Saved spiral: X{X.shape}, y{y.shape}')

        # 2. Generate XOR Dataset
        elif choice == '2':
            X, y = generate_xor(n_points=1000)
            torch.save({'X':X, 'y':y}, 'saved/xor.pt')
            print(f'Saved XOR: X{X.shape}, y{y.shape}')

        # 3. Train MLP from scratch
        elif choice == '3':
            data = torch.load('saved/spiral.pt')
            loader = create_dataloader(data['X'], data['y'], batch_size=64)
            model = MLP(2, 64, 32, 3).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(1, 101):
                loss = train_epoch(model, loader, optimizer, device)
                if epoch % 20 == 0:
                    acc = eval_model(model, loader, device)
                    print_loss(epoch, loss)
                    print_accuracy(acc)
            print('MLP training done')

        # 4. Train MLP with nn.Sequential
        elif choice == '4':
            data = torch.load('saved/xor.pt')
            loader = create_dataloader(data['X'], data['y'], batch_size=32)
            model = MLPSequential(2, 16, 8, 2).to(device)
            optimizer = SGDFullFromScratch(model.parameters(), lr=0.1, momentum=0.9)

            for epoch in range(1, 501):
                loss = train_epoch(model, loader, optimizer, device, clip_grad=True)
                if epoch % 100 == 0:
                    acc = eval_model(model, loader, device)
                    print(f'Epoch {epoch}: Loss {loss:.4f}, Acc {acc*100:.1f}%')
                    if acc == 1.0:
                        print('✅ XOR solved 100%')
                        break

        # 5. Train CNN on fake images
        elif choice == '5':
            X = torch.randn(2000, 3, 32, 32)
            y = torch.randint(0, 10, (2000,))
            loader = create_dataloader(X, y, batch_size=128)
            model = SimpleCNN(10).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(1, 21):
                loss = train_epoch(model, loader, optimizer, device)
                if epoch % 5 == 0:
                    acc = eval_model(model, loader, device)
                    print(f'Epoch {epoch}: Loss {loss:.4f}, Acc {acc*100:.1f}%')

        # 6. Save Model
        elif choice == '6':
            if model is None:
                print('Train a model first')
            else:
                save_model(model, 'saved/model.pt')
                print('Model saved to saved/model.pt')

        # 7. Load Model
        elif choice == '7':
            model = MLP(2, 64, 32, 3)
            model = load_model(model, 'saved/model.pt', device)
            print('Model loaded')

        # 8. Test LinearFromScratch vs nn.Linear
        elif choice == '8':
            test_linear_grad()

        # 9. Exit
        elif choice == '9':
            break
        else:
            print('Invalid choice')


    def test_linear_grad():
        """Challenge: Verify LinearFromScratch grads match Linear"""
        from models.layers import LinearFromScratch
        torch.manual_seed(0)
        x = torch.randn(10, 5)

        lin1 = LinearFromScratch(5, 3)
        lin2 = nn.Linear(5, 3)
        lin2.weight.data = lin1.weight.data.clone()
        lin2.bias.data = lin1.bias.data.clone()

        out1 = lin1(x).sum()
        out2 = lin2(x).sum()

        out1.backward()
        out2.backward()

        w_diff = torch.abs(lin1.weight.grad - lin2.weight.grad).max()
        b_diff = torch.abs(lin1.bias.grad - lin2.bias.grad).max()

        print(f'Max weight diff: {w_diff:.8f}')
        print(f'Max bias grad diff: {b_diff:.8f}')
        print('✅ Grads match!' if w_diff < 1e-6 else '❌ Mismatch')


if __name__ == '__main__':
    menu()