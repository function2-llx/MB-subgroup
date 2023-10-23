from luolib.lightning import LightningCLI

class CLI(LightningCLI):
    def __init__(self):
        super().__init__()

def main():
    CLI()

if __name__ == '__main__':
    main()
