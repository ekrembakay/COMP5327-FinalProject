import os

def load_data():
    X, Y = [], []
    input_path = os.path.join(os.getcwd(), "Input")
    output_path = os.path.join(os.getcwd(), "Output")

    for file in os.listdir(input_path):
        with open(os.path.join(input_path, file)) as f:
            X.append(f.read())
        with open(os.path.join(output_path, file)) as f:
            Y.append(f.read())
    return X, Y
if __name__ == '__main__':
    print(load_data())