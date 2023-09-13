import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':
    main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    device = torch.device('cuda2' if torch.cuda.is_available() else 'cpu')


    history_beta0 = pd.read_csv('history_beta0.csv')