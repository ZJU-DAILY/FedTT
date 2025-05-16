import argparse
from party.client import Client
from party.server import Server
import secretflow as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default='localhost')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--clients_num', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--frozen', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--source_dataset', type=list, default=['pemsd4', 'ft_aed', 'hk_traffic'])
    parser.add_argument('--target_dataset', type=str, default='pemsd8')
    parser.add_argument('--task', type=str, default='flow')
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--time_step', type=int, default=12)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--val_split', type=float, default=0.1)
    args = parser.parse_args()

    cs = ['client' + str(i) for i in range(args.clients_num)]
    sf.init(parties=['server'] + cs, address=args.address)
    clients = [Client(args=args, index=i, party=sf.SPU(party=cs[i])) for i in range(args.clients_num)]
    server = Server(args=args, party=sf.SPU(party='server'), clients=clients)
    server.init()
    for client in clients:
        client.server = server
        client.init()
    server.train()
